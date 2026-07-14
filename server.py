"""
EVM CFG → Solidity Decompiler — FastAPI backend.

Run:
    pip install fastapi uvicorn transformers torch
    python server.py
Then open http://localhost:8000
"""

import os
import sys
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
NOVA_DIR = os.path.join(ROOT, 'nova')
CKPT     = os.path.join(ROOT, 'checkpoints', 'nova-solidity-1.3b', 'checkpoint-24060')
sys.path.insert(0, NOVA_DIR)

from transformers import AutoTokenizer
from modeling_nova import NovaTokenizer, NovaForCausalLM
from prepare_solidity_dataset import normalize_cfg

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# ── Globals (loaded once at startup) ─────────────────────────────────────────
tokenizer = None
nova_tok  = None
model     = None


def load_model():
    global tokenizer, nova_tok, model
    print(f'[startup] device = {DEVICE}')
    print('[startup] loading tokenizer...')
    tok = AutoTokenizer.from_pretrained(
        'deepseek-ai/deepseek-coder-1.3b-base', trust_remote_code=True
    )
    tok.add_tokens(
        ['<unk>', '<cls>'] + [f'<label-{i}>' for i in range(1, 257)],
        special_tokens=True,
    )
    tok.pad_token    = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tokenizer = tok
    nova_tok  = NovaTokenizer(tok)

    print(f'[startup] loading model from {CKPT} ...')
    dtype = torch.bfloat16 if DEVICE == 'cuda' else torch.float32
    model = NovaForCausalLM.from_pretrained(
        CKPT, torch_dtype=dtype, trust_remote_code=True
    ).to(DEVICE).eval()
    print('[startup] model ready!')


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title='EVM Decompiler', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── API schema ────────────────────────────────────────────────────────────────
class DecompileRequest(BaseModel):
    cfg: str           = ''   # pre-built CFG text (optional if bytecode given)
    bytecode: str      = ''   # raw hex bytecode (optional if cfg given)
    func_name: str     = ''   # which function to extract (bytecode mode only)
    version: str       = '0.8.9'
    max_tokens: int    = 512

class DecompileResponse(BaseModel):
    solidity: str
    input_tokens: int
    func_names: list[str] = []   # populated in bytecode mode

class ListFunctionsRequest(BaseModel):
    bytecode: str

class ListFunctionsResponse(BaseModel):
    functions: list[str]
    cfgs: dict[str, str]    # name → cfg_text


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(cfg_text: str, version: str, max_new_tokens: int) -> tuple[str, int]:
    version   = version.strip() or '0.8.x'
    cfg_norm  = normalize_cfg(cfg_text.strip())

    prompt_before = f'# This is the EVM CFG for a Solidity {version} function:\n'
    prompt_after  = '\nWhat is the Solidity source code?\n'
    input_text    = prompt_before + cfg_norm + prompt_after
    char_types    = (
        '0' * len(prompt_before) +
        '1' * len(cfg_norm) +
        '0' * len(prompt_after)
    )

    enc         = nova_tok.encode(input_text, '', char_types)
    input_ids   = torch.LongTensor([enc['input_ids'].tolist()]).to(DEVICE)
    nova_mask   = torch.LongTensor(enc['nova_attention_mask']).unsqueeze(0).to(DEVICE)
    no_mask_idx = torch.LongTensor([enc['no_mask_idx']]).to(DEVICE)

    outputs = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        nova_attention_mask=nova_mask,
        no_mask_idx=no_mask_idx,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(
        outputs[0][input_ids.size(1):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return decoded.strip(), input_ids.size(1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post('/api/list_functions', response_model=ListFunctionsResponse)
async def list_functions(req: ListFunctionsRequest):
    """Return all function names found in the bytecode (no inference)."""
    if not req.bytecode.strip():
        raise HTTPException(status_code=400, detail='bytecode is empty.')
    try:
        from bytecode_to_cfg import extract_function_cfgs
        cfgs = extract_function_cfgs(req.bytecode)
        return ListFunctionsResponse(functions=list(cfgs.keys()), cfgs=cfgs)
    except ImportError:
        raise HTTPException(status_code=500, detail='evm_cfg_builder not installed. Run: pip install evm_cfg_builder')
    except Exception as e:
        import traceback
        print(f"\n[ERROR] list_functions failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/api/decompile', response_model=DecompileResponse)
async def decompile(req: DecompileRequest):
    if req.max_tokens < 16 or req.max_tokens > 2048:
        raise HTTPException(status_code=400, detail='max_tokens must be between 16 and 2048.')

    func_names: list[str] = []

    # ── Bytecode mode: extract CFG first ──────────────────────────────────────
    if req.bytecode.strip():
        try:
            from bytecode_to_cfg import extract_function_cfgs, bytecode_to_cfg_text
            cfgs = extract_function_cfgs(req.bytecode)
            func_names = list(cfgs.keys())
            if not cfgs:
                raise HTTPException(status_code=400, detail='No functions found in bytecode.')
            if req.func_name and req.func_name in cfgs:
                cfg_text = cfgs[req.func_name]
            else:
                # Use first non-dispatcher function
                skip = {"fallback", "_fallback", "dispatcher"}
                cfg_text = next((v for k, v in cfgs.items() if k not in skip), next(iter(cfgs.values())))
        except ImportError:
            raise HTTPException(status_code=500, detail='evm_cfg_builder not installed. Run: pip install evm_cfg_builder')
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'CFG extraction failed: {e}')

    # ── CFG mode: use provided CFG directly ───────────────────────────────────
    elif req.cfg.strip():
        cfg_text = req.cfg

    else:
        raise HTTPException(status_code=400, detail='Provide either cfg or bytecode.')

    from asyncio import get_event_loop
    from concurrent.futures import ThreadPoolExecutor
    loop = get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        solidity, n_input = await loop.run_in_executor(
            pool, run_inference, cfg_text, req.version, req.max_tokens
        )

    return DecompileResponse(solidity=solidity, input_tokens=n_input, func_names=func_names)


@app.get('/api/health')
def health():
    return {'status': 'ok', 'device': DEVICE}


# Serve frontend
app.mount('/', StaticFiles(directory=os.path.join(ROOT, 'static'), html=True), name='static')


if __name__ == '__main__':
    uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=False)
