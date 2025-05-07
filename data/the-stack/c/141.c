#pragma once
#include "IAudioPlayerView.h"

#include "HTMLayoutWindow.h"

class AudioPlayerView : public HTMLayoutWindow, public IAudioPlayerView
{
	MacroEvent(Cancel)
	MacroEvent(Hide)
	MacroEvent(Play)
	MacroEvent(Show)
	MacroEvent(Stop)

// data

private:

	HWND           parent;
	HINSTANCE      instance;
	SHACTIVATEINFO activateInfo;

// interface

public:

	AudioPlayerView
		( HINSTANCE         instance
		, bool              highRes
		, IHtmlDataLoader & htmlDataLoader
		);

	void Create(HWND parent);

	virtual void RegisterEventHandlers();

// IAudioPlayerView implementation

public:

	virtual void Hide();

	virtual void SetFileName(std::wstring & name);

	virtual void Show();

// windows message handlers

private:

	void OnActivate (Msg<WM_ACTIVATE> & msg);
	void OnClose    (Msg<WM_CLOSE>    & msg);
	void OnCommand  (Msg<WM_COMMAND>  & msg);
	void OnKeyUp    (Msg<WM_KEYUP>    & msg);

	virtual void ProcessMessage(WndMsg &msg);

// HTMLayout message handlers

private:

	void OnVoicePlay(BEHAVIOR_EVENT_PARAMS * params);
	void OnVoiceStop(BEHAVIOR_EVENT_PARAMS * params);

// utility functions

private:

	ATOM RegisterClass(const std::wstring & wndClass);

	void ResizeWindow();
};
