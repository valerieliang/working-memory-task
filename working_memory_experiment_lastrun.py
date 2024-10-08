#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.1),
    on October 08, 2024, at 10:16
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.1'
expName = 'working_memory_experiment'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\TheAn\\OneDrive\\Documents\\visual_working_memory_task\\working_memory_experiment_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file
    # return log file
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=(1024, 768), fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    text_norm = visual.TextStim(win=win, name='text_norm',
        text="Colored blocks will be shown on the screen.\nA certain location will be selected.\nUse the slider to indicate the color of the block that corresponds to that location.\n\nWhen you're ready, press [SPACE] to continue.",
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct = keyboard.Keyboard()
    # Run 'Begin Experiment' code from text_align
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    
    # --- Initialize components for Routine "trial" ---
    fixation = visual.ShapeStim(
        win=win, name='fixation', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    rect1 = visual.Rect(
        win=win, name='rect1',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    rect2 = visual.Rect(
        win=win, name='rect2',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    rect3 = visual.Rect(
        win=win, name='rect3',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    rect4 = visual.Rect(
        win=win, name='rect4',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    rect5 = visual.Rect(
        win=win, name='rect5',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    rect6 = visual.Rect(
        win=win, name='rect6',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    rect7 = visual.Rect(
        win=win, name='rect7',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-7.0, interpolate=True)
    rect8 = visual.Rect(
        win=win, name='rect8',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    probe = visual.Rect(
        win=win, name='probe',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-9.0, interpolate=True)
    slider = visual.Slider(win=win, name='slider',
        startValue=0.5, size=(1, 0.1), pos=(0, -0.4), units=win.units,
        labels=None, ticks=[{0,1}], granularity=0.0,
        style='slider', styleTweaks=(), opacity=0.0,
        labelColor=[0.0000, 0.0000, 0.0000], markerColor=[0.0000, 0.0000, 0.0000], lineColor=[0.0000, 0.0000, 0.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-10, readOnly=False)
    colorbar = visual.ImageStim(
        win=win,
        name='colorbar', 
        image='colorbar.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.4), size=(1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-11.0)
    
    # --- Initialize components for Routine "rest" ---
    text_norm_2 = visual.TextStim(win=win, name='text_norm_2',
        text="If you are ready for the next block, press 'space' to continue",
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_2 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from text_align_2
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    
    # --- Initialize components for Routine "trial" ---
    fixation = visual.ShapeStim(
        win=win, name='fixation', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    rect1 = visual.Rect(
        win=win, name='rect1',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    rect2 = visual.Rect(
        win=win, name='rect2',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    rect3 = visual.Rect(
        win=win, name='rect3',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    rect4 = visual.Rect(
        win=win, name='rect4',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    rect5 = visual.Rect(
        win=win, name='rect5',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    rect6 = visual.Rect(
        win=win, name='rect6',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    rect7 = visual.Rect(
        win=win, name='rect7',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-7.0, interpolate=True)
    rect8 = visual.Rect(
        win=win, name='rect8',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    probe = visual.Rect(
        win=win, name='probe',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-9.0, interpolate=True)
    slider = visual.Slider(win=win, name='slider',
        startValue=0.5, size=(1, 0.1), pos=(0, -0.4), units=win.units,
        labels=None, ticks=[{0,1}], granularity=0.0,
        style='slider', styleTweaks=(), opacity=0.0,
        labelColor=[0.0000, 0.0000, 0.0000], markerColor=[0.0000, 0.0000, 0.0000], lineColor=[0.0000, 0.0000, 0.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-10, readOnly=False)
    colorbar = visual.ImageStim(
        win=win,
        name='colorbar', 
        image='colorbar.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.4), size=(1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-11.0)
    
    # --- Initialize components for Routine "rest" ---
    text_norm_2 = visual.TextStim(win=win, name='text_norm_2',
        text="If you are ready for the next block, press 'space' to continue",
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_2 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from text_align_2
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    
    # --- Initialize components for Routine "trial" ---
    fixation = visual.ShapeStim(
        win=win, name='fixation', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    rect1 = visual.Rect(
        win=win, name='rect1',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    rect2 = visual.Rect(
        win=win, name='rect2',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    rect3 = visual.Rect(
        win=win, name='rect3',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    rect4 = visual.Rect(
        win=win, name='rect4',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    rect5 = visual.Rect(
        win=win, name='rect5',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    rect6 = visual.Rect(
        win=win, name='rect6',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    rect7 = visual.Rect(
        win=win, name='rect7',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-7.0, interpolate=True)
    rect8 = visual.Rect(
        win=win, name='rect8',
        width=(0.05, 0.05)[0], height=(0.05, 0.05)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    probe = visual.Rect(
        win=win, name='probe',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-9.0, interpolate=True)
    slider = visual.Slider(win=win, name='slider',
        startValue=0.5, size=(1, 0.1), pos=(0, -0.4), units=win.units,
        labels=None, ticks=[{0,1}], granularity=0.0,
        style='slider', styleTweaks=(), opacity=0.0,
        labelColor=[0.0000, 0.0000, 0.0000], markerColor=[0.0000, 0.0000, 0.0000], lineColor=[0.0000, 0.0000, 0.0000], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-10, readOnly=False)
    colorbar = visual.ImageStim(
        win=win,
        name='colorbar', 
        image='colorbar.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.4), size=(1, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-11.0)
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions.started', globalClock.getTime())
    key_instruct.keys = []
    key_instruct.rt = []
    _key_instruct_allKeys = []
    # keep track of which components have finished
    instructionsComponents = [text_norm, key_instruct]
    for thisComponent in instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm* updates
        
        # if text_norm is starting this frame...
        if text_norm.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm.frameNStart = frameN  # exact frame index
            text_norm.tStart = t  # local t and not account for scr refresh
            text_norm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm.status = STARTED
            text_norm.setAutoDraw(True)
        
        # if text_norm is active this frame...
        if text_norm.status == STARTED:
            # update params
            pass
        
        # *key_instruct* updates
        waitOnFlip = False
        
        # if key_instruct is starting this frame...
        if key_instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct.frameNStart = frameN  # exact frame index
            key_instruct.tStart = t  # local t and not account for scr refresh
            key_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct.started')
            # update status
            key_instruct.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_allKeys.extend(theseKeys)
            if len(_key_instruct_allKeys):
                key_instruct.keys = _key_instruct_allKeys[0].name  # just the first key pressed
                key_instruct.rt = _key_instruct_allKeys[0].rt
                key_instruct.duration = _key_instruct_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions.stopped', globalClock.getTime())
    # check responses
    if key_instruct.keys in ['', [], None]:  # No response was made
        key_instruct.keys = None
    thisExp.addData('key_instruct.keys',key_instruct.keys)
    if key_instruct.keys != None:  # we had a response
        thisExp.addData('key_instruct.rt', key_instruct.rt)
        thisExp.addData('key_instruct.duration', key_instruct.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Block1 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('C:/Users/TheAn/OneDrive/Documents/visual_working_memory_task/WorkingMemory_block1.xlsx'),
        seed=None, name='Block1')
    thisExp.addLoop(Block1)  # add the loop to the experiment
    thisBlock1 = Block1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock1.rgb)
    if thisBlock1 != None:
        for paramName in thisBlock1:
            globals()[paramName] = thisBlock1[paramName]
    
    for thisBlock1 in Block1:
        currentLoop = Block1
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock1.rgb)
        if thisBlock1 != None:
            for paramName in thisBlock1:
                globals()[paramName] = thisBlock1[paramName]
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime())
        rect1.setFillColor(rect1_col)
        rect1.setPos(rect1_loc)
        rect2.setFillColor(rect2_col)
        rect2.setPos(rect2_loc)
        rect3.setFillColor(rect3_col)
        rect3.setPos(rect3_loc)
        rect4.setFillColor(rect4_col)
        rect4.setPos(rect4_loc)
        rect5.setFillColor(rect5_col)
        rect5.setPos(rect5_loc)
        rect6.setFillColor(rect6_col)
        rect6.setPos(rect6_loc)
        rect7.setFillColor(rect7_col)
        rect7.setPos(rect7_loc)
        rect8.setFillColor(rect8_col)
        rect8.setPos(rect8_loc)
        probe.setPos(probe_loc)
        slider.reset()
        # keep track of which components have finished
        trialComponents = [fixation, rect1, rect2, rect3, rect4, rect5, rect6, rect7, rect8, probe, slider, colorbar]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *rect1* updates
            
            # if rect1 is starting this frame...
            if rect1.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect1.frameNStart = frameN  # exact frame index
                rect1.tStart = t  # local t and not account for scr refresh
                rect1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect1.started')
                # update status
                rect1.status = STARTED
                rect1.setAutoDraw(True)
            
            # if rect1 is active this frame...
            if rect1.status == STARTED:
                # update params
                pass
            
            # if rect1 is stopping this frame...
            if rect1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect1.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect1.tStop = t  # not accounting for scr refresh
                    rect1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect1.stopped')
                    # update status
                    rect1.status = FINISHED
                    rect1.setAutoDraw(False)
            
            # *rect2* updates
            
            # if rect2 is starting this frame...
            if rect2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect2.frameNStart = frameN  # exact frame index
                rect2.tStart = t  # local t and not account for scr refresh
                rect2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect2.started')
                # update status
                rect2.status = STARTED
                rect2.setAutoDraw(True)
            
            # if rect2 is active this frame...
            if rect2.status == STARTED:
                # update params
                pass
            
            # if rect2 is stopping this frame...
            if rect2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect2.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect2.tStop = t  # not accounting for scr refresh
                    rect2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect2.stopped')
                    # update status
                    rect2.status = FINISHED
                    rect2.setAutoDraw(False)
            
            # *rect3* updates
            
            # if rect3 is starting this frame...
            if rect3.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect3.frameNStart = frameN  # exact frame index
                rect3.tStart = t  # local t and not account for scr refresh
                rect3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect3.started')
                # update status
                rect3.status = STARTED
                rect3.setAutoDraw(True)
            
            # if rect3 is active this frame...
            if rect3.status == STARTED:
                # update params
                pass
            
            # if rect3 is stopping this frame...
            if rect3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect3.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect3.tStop = t  # not accounting for scr refresh
                    rect3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect3.stopped')
                    # update status
                    rect3.status = FINISHED
                    rect3.setAutoDraw(False)
            
            # *rect4* updates
            
            # if rect4 is starting this frame...
            if rect4.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect4.frameNStart = frameN  # exact frame index
                rect4.tStart = t  # local t and not account for scr refresh
                rect4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect4.started')
                # update status
                rect4.status = STARTED
                rect4.setAutoDraw(True)
            
            # if rect4 is active this frame...
            if rect4.status == STARTED:
                # update params
                pass
            
            # if rect4 is stopping this frame...
            if rect4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect4.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect4.tStop = t  # not accounting for scr refresh
                    rect4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect4.stopped')
                    # update status
                    rect4.status = FINISHED
                    rect4.setAutoDraw(False)
            
            # *rect5* updates
            
            # if rect5 is starting this frame...
            if rect5.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect5.frameNStart = frameN  # exact frame index
                rect5.tStart = t  # local t and not account for scr refresh
                rect5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect5.started')
                # update status
                rect5.status = STARTED
                rect5.setAutoDraw(True)
            
            # if rect5 is active this frame...
            if rect5.status == STARTED:
                # update params
                pass
            
            # if rect5 is stopping this frame...
            if rect5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect5.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect5.tStop = t  # not accounting for scr refresh
                    rect5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect5.stopped')
                    # update status
                    rect5.status = FINISHED
                    rect5.setAutoDraw(False)
            
            # *rect6* updates
            
            # if rect6 is starting this frame...
            if rect6.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect6.frameNStart = frameN  # exact frame index
                rect6.tStart = t  # local t and not account for scr refresh
                rect6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect6.started')
                # update status
                rect6.status = STARTED
                rect6.setAutoDraw(True)
            
            # if rect6 is active this frame...
            if rect6.status == STARTED:
                # update params
                pass
            
            # if rect6 is stopping this frame...
            if rect6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect6.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect6.tStop = t  # not accounting for scr refresh
                    rect6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect6.stopped')
                    # update status
                    rect6.status = FINISHED
                    rect6.setAutoDraw(False)
            
            # *rect7* updates
            
            # if rect7 is starting this frame...
            if rect7.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect7.frameNStart = frameN  # exact frame index
                rect7.tStart = t  # local t and not account for scr refresh
                rect7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect7.started')
                # update status
                rect7.status = STARTED
                rect7.setAutoDraw(True)
            
            # if rect7 is active this frame...
            if rect7.status == STARTED:
                # update params
                pass
            
            # if rect7 is stopping this frame...
            if rect7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect7.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect7.tStop = t  # not accounting for scr refresh
                    rect7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect7.stopped')
                    # update status
                    rect7.status = FINISHED
                    rect7.setAutoDraw(False)
            
            # *rect8* updates
            
            # if rect8 is starting this frame...
            if rect8.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect8.frameNStart = frameN  # exact frame index
                rect8.tStart = t  # local t and not account for scr refresh
                rect8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect8.started')
                # update status
                rect8.status = STARTED
                rect8.setAutoDraw(True)
            
            # if rect8 is active this frame...
            if rect8.status == STARTED:
                # update params
                pass
            
            # if rect8 is stopping this frame...
            if rect8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect8.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect8.tStop = t  # not accounting for scr refresh
                    rect8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect8.stopped')
                    # update status
                    rect8.status = FINISHED
                    rect8.setAutoDraw(False)
            
            # *probe* updates
            
            # if probe is starting this frame...
            if probe.status == NOT_STARTED and tThisFlip >= 2.4-frameTolerance:
                # keep track of start time/frame for later
                probe.frameNStart = frameN  # exact frame index
                probe.tStart = t  # local t and not account for scr refresh
                probe.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe.started')
                # update status
                probe.status = STARTED
                probe.setAutoDraw(True)
            
            # if probe is active this frame...
            if probe.status == STARTED:
                # update params
                pass
            
            # *slider* updates
            
            # if slider is starting this frame...
            if slider.status == NOT_STARTED and tThisFlip >= 2.4-frameTolerance:
                # keep track of start time/frame for later
                slider.frameNStart = frameN  # exact frame index
                slider.tStart = t  # local t and not account for scr refresh
                slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider.started')
                # update status
                slider.status = STARTED
                slider.setAutoDraw(True)
            
            # if slider is active this frame...
            if slider.status == STARTED:
                # update params
                pass
            
            # Check slider for response to end Routine
            if slider.getRating() is not None and slider.status == STARTED:
                continueRoutine = False
            
            # *colorbar* updates
            
            # if colorbar is starting this frame...
            if colorbar.status == NOT_STARTED and tThisFlip >= 2.4-frameTolerance:
                # keep track of start time/frame for later
                colorbar.frameNStart = frameN  # exact frame index
                colorbar.tStart = t  # local t and not account for scr refresh
                colorbar.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(colorbar, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'colorbar.started')
                # update status
                colorbar.status = STARTED
                colorbar.setAutoDraw(True)
            
            # if colorbar is active this frame...
            if colorbar.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime())
        Block1.addData('slider.response', slider.getRating())
        Block1.addData('slider.rt', slider.getRT())
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'Block1'
    
    
    # --- Prepare to start Routine "rest" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('rest.started', globalClock.getTime())
    key_instruct_2.keys = []
    key_instruct_2.rt = []
    _key_instruct_2_allKeys = []
    # keep track of which components have finished
    restComponents = [text_norm_2, key_instruct_2]
    for thisComponent in restComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "rest" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm_2* updates
        
        # if text_norm_2 is starting this frame...
        if text_norm_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm_2.frameNStart = frameN  # exact frame index
            text_norm_2.tStart = t  # local t and not account for scr refresh
            text_norm_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm_2.status = STARTED
            text_norm_2.setAutoDraw(True)
        
        # if text_norm_2 is active this frame...
        if text_norm_2.status == STARTED:
            # update params
            pass
        
        # *key_instruct_2* updates
        waitOnFlip = False
        
        # if key_instruct_2 is starting this frame...
        if key_instruct_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_2.frameNStart = frameN  # exact frame index
            key_instruct_2.tStart = t  # local t and not account for scr refresh
            key_instruct_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct_2.started')
            # update status
            key_instruct_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_2.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_2_allKeys.extend(theseKeys)
            if len(_key_instruct_2_allKeys):
                key_instruct_2.keys = _key_instruct_2_allKeys[0].name  # just the first key pressed
                key_instruct_2.rt = _key_instruct_2_allKeys[0].rt
                key_instruct_2.duration = _key_instruct_2_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in restComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "rest" ---
    for thisComponent in restComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('rest.stopped', globalClock.getTime())
    # check responses
    if key_instruct_2.keys in ['', [], None]:  # No response was made
        key_instruct_2.keys = None
    thisExp.addData('key_instruct_2.keys',key_instruct_2.keys)
    if key_instruct_2.keys != None:  # we had a response
        thisExp.addData('key_instruct_2.rt', key_instruct_2.rt)
        thisExp.addData('key_instruct_2.duration', key_instruct_2.duration)
    thisExp.nextEntry()
    # the Routine "rest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Block2 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('WorkingMemory_block2.xlsx'),
        seed=None, name='Block2')
    thisExp.addLoop(Block2)  # add the loop to the experiment
    thisBlock2 = Block2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock2.rgb)
    if thisBlock2 != None:
        for paramName in thisBlock2:
            globals()[paramName] = thisBlock2[paramName]
    
    for thisBlock2 in Block2:
        currentLoop = Block2
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock2.rgb)
        if thisBlock2 != None:
            for paramName in thisBlock2:
                globals()[paramName] = thisBlock2[paramName]
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime())
        rect1.setFillColor(rect1_col)
        rect1.setPos(rect1_loc)
        rect2.setFillColor(rect2_col)
        rect2.setPos(rect2_loc)
        rect3.setFillColor(rect3_col)
        rect3.setPos(rect3_loc)
        rect4.setFillColor(rect4_col)
        rect4.setPos(rect4_loc)
        rect5.setFillColor(rect5_col)
        rect5.setPos(rect5_loc)
        rect6.setFillColor(rect6_col)
        rect6.setPos(rect6_loc)
        rect7.setFillColor(rect7_col)
        rect7.setPos(rect7_loc)
        rect8.setFillColor(rect8_col)
        rect8.setPos(rect8_loc)
        probe.setPos(probe_loc)
        slider.reset()
        # keep track of which components have finished
        trialComponents = [fixation, rect1, rect2, rect3, rect4, rect5, rect6, rect7, rect8, probe, slider, colorbar]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *rect1* updates
            
            # if rect1 is starting this frame...
            if rect1.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect1.frameNStart = frameN  # exact frame index
                rect1.tStart = t  # local t and not account for scr refresh
                rect1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect1.started')
                # update status
                rect1.status = STARTED
                rect1.setAutoDraw(True)
            
            # if rect1 is active this frame...
            if rect1.status == STARTED:
                # update params
                pass
            
            # if rect1 is stopping this frame...
            if rect1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect1.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect1.tStop = t  # not accounting for scr refresh
                    rect1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect1.stopped')
                    # update status
                    rect1.status = FINISHED
                    rect1.setAutoDraw(False)
            
            # *rect2* updates
            
            # if rect2 is starting this frame...
            if rect2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect2.frameNStart = frameN  # exact frame index
                rect2.tStart = t  # local t and not account for scr refresh
                rect2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect2.started')
                # update status
                rect2.status = STARTED
                rect2.setAutoDraw(True)
            
            # if rect2 is active this frame...
            if rect2.status == STARTED:
                # update params
                pass
            
            # if rect2 is stopping this frame...
            if rect2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect2.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect2.tStop = t  # not accounting for scr refresh
                    rect2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect2.stopped')
                    # update status
                    rect2.status = FINISHED
                    rect2.setAutoDraw(False)
            
            # *rect3* updates
            
            # if rect3 is starting this frame...
            if rect3.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect3.frameNStart = frameN  # exact frame index
                rect3.tStart = t  # local t and not account for scr refresh
                rect3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect3.started')
                # update status
                rect3.status = STARTED
                rect3.setAutoDraw(True)
            
            # if rect3 is active this frame...
            if rect3.status == STARTED:
                # update params
                pass
            
            # if rect3 is stopping this frame...
            if rect3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect3.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect3.tStop = t  # not accounting for scr refresh
                    rect3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect3.stopped')
                    # update status
                    rect3.status = FINISHED
                    rect3.setAutoDraw(False)
            
            # *rect4* updates
            
            # if rect4 is starting this frame...
            if rect4.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect4.frameNStart = frameN  # exact frame index
                rect4.tStart = t  # local t and not account for scr refresh
                rect4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect4.started')
                # update status
                rect4.status = STARTED
                rect4.setAutoDraw(True)
            
            # if rect4 is active this frame...
            if rect4.status == STARTED:
                # update params
                pass
            
            # if rect4 is stopping this frame...
            if rect4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect4.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect4.tStop = t  # not accounting for scr refresh
                    rect4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect4.stopped')
                    # update status
                    rect4.status = FINISHED
                    rect4.setAutoDraw(False)
            
            # *rect5* updates
            
            # if rect5 is starting this frame...
            if rect5.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect5.frameNStart = frameN  # exact frame index
                rect5.tStart = t  # local t and not account for scr refresh
                rect5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect5.started')
                # update status
                rect5.status = STARTED
                rect5.setAutoDraw(True)
            
            # if rect5 is active this frame...
            if rect5.status == STARTED:
                # update params
                pass
            
            # if rect5 is stopping this frame...
            if rect5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect5.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect5.tStop = t  # not accounting for scr refresh
                    rect5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect5.stopped')
                    # update status
                    rect5.status = FINISHED
                    rect5.setAutoDraw(False)
            
            # *rect6* updates
            
            # if rect6 is starting this frame...
            if rect6.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect6.frameNStart = frameN  # exact frame index
                rect6.tStart = t  # local t and not account for scr refresh
                rect6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect6.started')
                # update status
                rect6.status = STARTED
                rect6.setAutoDraw(True)
            
            # if rect6 is active this frame...
            if rect6.status == STARTED:
                # update params
                pass
            
            # if rect6 is stopping this frame...
            if rect6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect6.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect6.tStop = t  # not accounting for scr refresh
                    rect6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect6.stopped')
                    # update status
                    rect6.status = FINISHED
                    rect6.setAutoDraw(False)
            
            # *rect7* updates
            
            # if rect7 is starting this frame...
            if rect7.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect7.frameNStart = frameN  # exact frame index
                rect7.tStart = t  # local t and not account for scr refresh
                rect7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect7.started')
                # update status
                rect7.status = STARTED
                rect7.setAutoDraw(True)
            
            # if rect7 is active this frame...
            if rect7.status == STARTED:
                # update params
                pass
            
            # if rect7 is stopping this frame...
            if rect7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect7.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect7.tStop = t  # not accounting for scr refresh
                    rect7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect7.stopped')
                    # update status
                    rect7.status = FINISHED
                    rect7.setAutoDraw(False)
            
            # *rect8* updates
            
            # if rect8 is starting this frame...
            if rect8.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect8.frameNStart = frameN  # exact frame index
                rect8.tStart = t  # local t and not account for scr refresh
                rect8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect8.started')
                # update status
                rect8.status = STARTED
                rect8.setAutoDraw(True)
            
            # if rect8 is active this frame...
            if rect8.status == STARTED:
                # update params
                pass
            
            # if rect8 is stopping this frame...
            if rect8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect8.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect8.tStop = t  # not accounting for scr refresh
                    rect8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect8.stopped')
                    # update status
                    rect8.status = FINISHED
                    rect8.setAutoDraw(False)
            
            # *probe* updates
            
            # if probe is starting this frame...
            if probe.status == NOT_STARTED and tThisFlip >= 2.4-frameTolerance:
                # keep track of start time/frame for later
                probe.frameNStart = frameN  # exact frame index
                probe.tStart = t  # local t and not account for scr refresh
                probe.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe.started')
                # update status
                probe.status = STARTED
                probe.setAutoDraw(True)
            
            # if probe is active this frame...
            if probe.status == STARTED:
                # update params
                pass
            
            # *slider* updates
            
            # if slider is starting this frame...
            if slider.status == NOT_STARTED and tThisFlip >= 2.4-frameTolerance:
                # keep track of start time/frame for later
                slider.frameNStart = frameN  # exact frame index
                slider.tStart = t  # local t and not account for scr refresh
                slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider.started')
                # update status
                slider.status = STARTED
                slider.setAutoDraw(True)
            
            # if slider is active this frame...
            if slider.status == STARTED:
                # update params
                pass
            
            # Check slider for response to end Routine
            if slider.getRating() is not None and slider.status == STARTED:
                continueRoutine = False
            
            # *colorbar* updates
            
            # if colorbar is starting this frame...
            if colorbar.status == NOT_STARTED and tThisFlip >= 2.4-frameTolerance:
                # keep track of start time/frame for later
                colorbar.frameNStart = frameN  # exact frame index
                colorbar.tStart = t  # local t and not account for scr refresh
                colorbar.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(colorbar, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'colorbar.started')
                # update status
                colorbar.status = STARTED
                colorbar.setAutoDraw(True)
            
            # if colorbar is active this frame...
            if colorbar.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime())
        Block2.addData('slider.response', slider.getRating())
        Block2.addData('slider.rt', slider.getRT())
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'Block2'
    
    
    # --- Prepare to start Routine "rest" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('rest.started', globalClock.getTime())
    key_instruct_2.keys = []
    key_instruct_2.rt = []
    _key_instruct_2_allKeys = []
    # keep track of which components have finished
    restComponents = [text_norm_2, key_instruct_2]
    for thisComponent in restComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "rest" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm_2* updates
        
        # if text_norm_2 is starting this frame...
        if text_norm_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm_2.frameNStart = frameN  # exact frame index
            text_norm_2.tStart = t  # local t and not account for scr refresh
            text_norm_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm_2.status = STARTED
            text_norm_2.setAutoDraw(True)
        
        # if text_norm_2 is active this frame...
        if text_norm_2.status == STARTED:
            # update params
            pass
        
        # *key_instruct_2* updates
        waitOnFlip = False
        
        # if key_instruct_2 is starting this frame...
        if key_instruct_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_2.frameNStart = frameN  # exact frame index
            key_instruct_2.tStart = t  # local t and not account for scr refresh
            key_instruct_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct_2.started')
            # update status
            key_instruct_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_2.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_2_allKeys.extend(theseKeys)
            if len(_key_instruct_2_allKeys):
                key_instruct_2.keys = _key_instruct_2_allKeys[0].name  # just the first key pressed
                key_instruct_2.rt = _key_instruct_2_allKeys[0].rt
                key_instruct_2.duration = _key_instruct_2_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in restComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "rest" ---
    for thisComponent in restComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('rest.stopped', globalClock.getTime())
    # check responses
    if key_instruct_2.keys in ['', [], None]:  # No response was made
        key_instruct_2.keys = None
    thisExp.addData('key_instruct_2.keys',key_instruct_2.keys)
    if key_instruct_2.keys != None:  # we had a response
        thisExp.addData('key_instruct_2.rt', key_instruct_2.rt)
        thisExp.addData('key_instruct_2.duration', key_instruct_2.duration)
    thisExp.nextEntry()
    # the Routine "rest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Block3 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('WorkingMemory_block3.xlsx'),
        seed=None, name='Block3')
    thisExp.addLoop(Block3)  # add the loop to the experiment
    thisBlock3 = Block3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock3.rgb)
    if thisBlock3 != None:
        for paramName in thisBlock3:
            globals()[paramName] = thisBlock3[paramName]
    
    for thisBlock3 in Block3:
        currentLoop = Block3
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock3.rgb)
        if thisBlock3 != None:
            for paramName in thisBlock3:
                globals()[paramName] = thisBlock3[paramName]
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime())
        rect1.setFillColor(rect1_col)
        rect1.setPos(rect1_loc)
        rect2.setFillColor(rect2_col)
        rect2.setPos(rect2_loc)
        rect3.setFillColor(rect3_col)
        rect3.setPos(rect3_loc)
        rect4.setFillColor(rect4_col)
        rect4.setPos(rect4_loc)
        rect5.setFillColor(rect5_col)
        rect5.setPos(rect5_loc)
        rect6.setFillColor(rect6_col)
        rect6.setPos(rect6_loc)
        rect7.setFillColor(rect7_col)
        rect7.setPos(rect7_loc)
        rect8.setFillColor(rect8_col)
        rect8.setPos(rect8_loc)
        probe.setPos(probe_loc)
        slider.reset()
        # keep track of which components have finished
        trialComponents = [fixation, rect1, rect2, rect3, rect4, rect5, rect6, rect7, rect8, probe, slider, colorbar]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *rect1* updates
            
            # if rect1 is starting this frame...
            if rect1.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect1.frameNStart = frameN  # exact frame index
                rect1.tStart = t  # local t and not account for scr refresh
                rect1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect1.started')
                # update status
                rect1.status = STARTED
                rect1.setAutoDraw(True)
            
            # if rect1 is active this frame...
            if rect1.status == STARTED:
                # update params
                pass
            
            # if rect1 is stopping this frame...
            if rect1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect1.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect1.tStop = t  # not accounting for scr refresh
                    rect1.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect1.stopped')
                    # update status
                    rect1.status = FINISHED
                    rect1.setAutoDraw(False)
            
            # *rect2* updates
            
            # if rect2 is starting this frame...
            if rect2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect2.frameNStart = frameN  # exact frame index
                rect2.tStart = t  # local t and not account for scr refresh
                rect2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect2.started')
                # update status
                rect2.status = STARTED
                rect2.setAutoDraw(True)
            
            # if rect2 is active this frame...
            if rect2.status == STARTED:
                # update params
                pass
            
            # if rect2 is stopping this frame...
            if rect2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect2.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect2.tStop = t  # not accounting for scr refresh
                    rect2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect2.stopped')
                    # update status
                    rect2.status = FINISHED
                    rect2.setAutoDraw(False)
            
            # *rect3* updates
            
            # if rect3 is starting this frame...
            if rect3.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect3.frameNStart = frameN  # exact frame index
                rect3.tStart = t  # local t and not account for scr refresh
                rect3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect3.started')
                # update status
                rect3.status = STARTED
                rect3.setAutoDraw(True)
            
            # if rect3 is active this frame...
            if rect3.status == STARTED:
                # update params
                pass
            
            # if rect3 is stopping this frame...
            if rect3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect3.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect3.tStop = t  # not accounting for scr refresh
                    rect3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect3.stopped')
                    # update status
                    rect3.status = FINISHED
                    rect3.setAutoDraw(False)
            
            # *rect4* updates
            
            # if rect4 is starting this frame...
            if rect4.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect4.frameNStart = frameN  # exact frame index
                rect4.tStart = t  # local t and not account for scr refresh
                rect4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect4.started')
                # update status
                rect4.status = STARTED
                rect4.setAutoDraw(True)
            
            # if rect4 is active this frame...
            if rect4.status == STARTED:
                # update params
                pass
            
            # if rect4 is stopping this frame...
            if rect4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect4.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect4.tStop = t  # not accounting for scr refresh
                    rect4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect4.stopped')
                    # update status
                    rect4.status = FINISHED
                    rect4.setAutoDraw(False)
            
            # *rect5* updates
            
            # if rect5 is starting this frame...
            if rect5.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect5.frameNStart = frameN  # exact frame index
                rect5.tStart = t  # local t and not account for scr refresh
                rect5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect5.started')
                # update status
                rect5.status = STARTED
                rect5.setAutoDraw(True)
            
            # if rect5 is active this frame...
            if rect5.status == STARTED:
                # update params
                pass
            
            # if rect5 is stopping this frame...
            if rect5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect5.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect5.tStop = t  # not accounting for scr refresh
                    rect5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect5.stopped')
                    # update status
                    rect5.status = FINISHED
                    rect5.setAutoDraw(False)
            
            # *rect6* updates
            
            # if rect6 is starting this frame...
            if rect6.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect6.frameNStart = frameN  # exact frame index
                rect6.tStart = t  # local t and not account for scr refresh
                rect6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect6.started')
                # update status
                rect6.status = STARTED
                rect6.setAutoDraw(True)
            
            # if rect6 is active this frame...
            if rect6.status == STARTED:
                # update params
                pass
            
            # if rect6 is stopping this frame...
            if rect6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect6.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect6.tStop = t  # not accounting for scr refresh
                    rect6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect6.stopped')
                    # update status
                    rect6.status = FINISHED
                    rect6.setAutoDraw(False)
            
            # *rect7* updates
            
            # if rect7 is starting this frame...
            if rect7.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect7.frameNStart = frameN  # exact frame index
                rect7.tStart = t  # local t and not account for scr refresh
                rect7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect7.started')
                # update status
                rect7.status = STARTED
                rect7.setAutoDraw(True)
            
            # if rect7 is active this frame...
            if rect7.status == STARTED:
                # update params
                pass
            
            # if rect7 is stopping this frame...
            if rect7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect7.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect7.tStop = t  # not accounting for scr refresh
                    rect7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect7.stopped')
                    # update status
                    rect7.status = FINISHED
                    rect7.setAutoDraw(False)
            
            # *rect8* updates
            
            # if rect8 is starting this frame...
            if rect8.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                rect8.frameNStart = frameN  # exact frame index
                rect8.tStart = t  # local t and not account for scr refresh
                rect8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rect8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect8.started')
                # update status
                rect8.status = STARTED
                rect8.setAutoDraw(True)
            
            # if rect8 is active this frame...
            if rect8.status == STARTED:
                # update params
                pass
            
            # if rect8 is stopping this frame...
            if rect8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rect8.tStartRefresh + .5-frameTolerance:
                    # keep track of stop time/frame for later
                    rect8.tStop = t  # not accounting for scr refresh
                    rect8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rect8.stopped')
                    # update status
                    rect8.status = FINISHED
                    rect8.setAutoDraw(False)
            
            # *probe* updates
            
            # if probe is starting this frame...
            if probe.status == NOT_STARTED and tThisFlip >= 2.4-frameTolerance:
                # keep track of start time/frame for later
                probe.frameNStart = frameN  # exact frame index
                probe.tStart = t  # local t and not account for scr refresh
                probe.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe.started')
                # update status
                probe.status = STARTED
                probe.setAutoDraw(True)
            
            # if probe is active this frame...
            if probe.status == STARTED:
                # update params
                pass
            
            # *slider* updates
            
            # if slider is starting this frame...
            if slider.status == NOT_STARTED and tThisFlip >= 2.4-frameTolerance:
                # keep track of start time/frame for later
                slider.frameNStart = frameN  # exact frame index
                slider.tStart = t  # local t and not account for scr refresh
                slider.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider.started')
                # update status
                slider.status = STARTED
                slider.setAutoDraw(True)
            
            # if slider is active this frame...
            if slider.status == STARTED:
                # update params
                pass
            
            # Check slider for response to end Routine
            if slider.getRating() is not None and slider.status == STARTED:
                continueRoutine = False
            
            # *colorbar* updates
            
            # if colorbar is starting this frame...
            if colorbar.status == NOT_STARTED and tThisFlip >= 2.4-frameTolerance:
                # keep track of start time/frame for later
                colorbar.frameNStart = frameN  # exact frame index
                colorbar.tStart = t  # local t and not account for scr refresh
                colorbar.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(colorbar, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'colorbar.started')
                # update status
                colorbar.status = STARTED
                colorbar.setAutoDraw(True)
            
            # if colorbar is active this frame...
            if colorbar.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime())
        Block3.addData('slider.response', slider.getRating())
        Block3.addData('slider.rt', slider.getRT())
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'Block3'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            eyetracker.setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
