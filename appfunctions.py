#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 20:14:01 2020

@author: clement
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from librosa.feature import melspectrogram
from librosa.core import power_to_db

def plot_audio(file):
    """
    Plot le signal temporel audio_data
    """
    audio_data, fe = librosa.load(file, sr=None)
    audio_data = audio_data[:10*fe]
    fig, ax = plt.subplots(figsize=(10,4))
    # Intervalle de temps entre deux points.
    dt = 1/fe 
    # Variable de temps en seconde.
    t = dt*np.arange(len(audio_data)) 
    ax.plot(t, audio_data)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig, audio_data.shape


def plot_Spectrogram(file):
    """
    Retourne le spectrogramme de y.
    La fréquence est tracée linéairement.
    """
    y, fe = librosa.load(file, sr=None)
    y = y[:10*fe]
    D_highres = librosa.stft(y, hop_length=512, n_fft=1024)
    S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
    fig, ax = plt.subplots(figsize=(10,4))
    img = librosa.display.specshow(S_db_hr, hop_length=512, sr=fe, 
                                   x_axis='time', y_axis='linear', ax=ax)
    plt.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set_xlim(0, 10)
    ax.set_ylim(20, fe//2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    return fig, S_db_hr.shape

def mfec(y=None, sr=22050, S=None, n_mfec=20, **kwargs):
    """Mel-frequency energy coefficients (MFECs)

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(d, t)] or None
        log-power Mel spectrogram

    n_mfcec: int > 0 [scalar]
        number of MFECs to return

    kwargs : additional keyword arguments
        Arguments to `melspectrogram`, if operating
        on time series input

    Returns
    -------
    M : np.ndarray [shape=(n_mfec, t)]
        MFEC sequence
    """
    if S is None:
        S = power_to_db(melspectrogram(y=y, sr=sr, power=1, **kwargs))

    M = S[:n_mfec]    
    return M


def plot_MFEC_Spectrogram(filename):
    """
    Trace le MFEC spectrogramme donné en entrée.
    La fréquence est tracée linéairement.
    """
    Tmax = 10
    audio, fs = librosa.load(filename, sr=None, res_type='kaiser_fast') 
    mfecs = mfec(y=audio[:Tmax*fs], sr=fs, n_mfec=128,
               n_fft=1024, hop_length=512)
    fig, ax = plt.subplots(figsize=(10,4))
    img = librosa.display.specshow(mfecs, sr=fs, x_axis='time', ax=ax,
                                   hop_length=512)
    plt.colorbar(img, ax=ax, format="%+2.f")
    ax.set_xlim(0, 10)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MFEC')
    return fig, mfecs.shape
