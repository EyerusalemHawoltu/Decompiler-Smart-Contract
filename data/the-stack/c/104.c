#include "packets.h"
#include "util.h"

#include "stdlib.h"

PACKET_FUNC_DEF(handshake)
{
    // TODO: rewrite server address. maybe?
    int32_t protocol_version;
    char* server_address = NULL;
    uint16_t server_port;
    int32_t next_state;

    bytebuf_read_varint(b, &protocol_version);
    bytebuf_read_string(b, &server_address);
    bytebuf_read_unsigned_short(b, &server_port);
    bytebuf_read_varint(b, &next_state);

    s->state = (enum estate)next_state;

    // put response here?
    
    free(server_address);
    return 0;
}

PACKET_FUNC_DEF(legacy_handshake)
{
    UNUSED(dir);
    UNUSED(b);
    // ignore incoming
    // 1.6 and below all send at least 0xFE
    // we are not required to send anything back
    bytebuf *out = bytebuf_new(0);
    stream_queue_client_add(s, out);
    return 0;
}
