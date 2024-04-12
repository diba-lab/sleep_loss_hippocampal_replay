from dataclasses import dataclass
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy.ndimage import gaussian_filter
from . import plotting
from .utils import signal_process


class Recinfo:
    """Parses .xml file to get a sense of recording parameters

    Attributes
    ----------
    basePath : str
        path to datafolder where .xml and .eeg files reside
    probemap : instance of Probemap class
        layout of recording channels
    sampfreq : int
        sampling rate of recording and is extracted from .xml file
    nChans : int,
        number of channels in the .dat/.eeg file
    channels : list
        list of recording channels in the .dat file from silicon probes and EXCLUDES any aux channels, skulleeg, emg, motion channels.
    channelgroups: list
        channels grouped in shanks.
    badchans : list
        list of bad channels
    skulleeg : list,
        channels in .dat/.eeg file from skull electrodes
    emgChans : list
        list of channels for emg
    nProbes : int,
        number of silicon probes used for this recording
    nShanksProbe : int or list of int,
        number of shanks in each probe. Example, [5,8], two probes with 5 and 8 shanks
    goodChans: list,
        list of channels excluding bad channels
    goodChangrp: list of lists,
        channels grouped in shanks and excludes bad channels. If all channels within a shank are bad, then it is represented as empty list within goodChangrp

    NOTE: len(channels) may not be equal to nChans.



    Methods
    ----------
    makerecinfo()
        creates a file containing basic infomation about the recording
    geteeg(chans, timeRange)
        returns lfp from .eeg file for given channels
    generate_xml(settingsPath)
        re-orders channels in the .xml file to reflect channel ordering in openephys settings.xml
    """

    def __init__(self, basePath):
        self.basePath = Path(basePath)

        filePrefix = None
        for file in os.listdir(basePath):
            if file.endswith(".xml"):
                xmlfile = self.basePath / file
                filePrefix = xmlfile.with_suffix("")

            elif file.endswith(".eeg"):
                eegfile = self.basePath / file
                filePrefix = eegfile.with_suffix("")
            elif file.endswith(".dat"):
                datfile = self.basePath / file
                filePrefix = datfile.with_suffix("")

        self.session = sessionname(filePrefix)
        self.files = files(filePrefix)
        self.recfiles = recfiles(filePrefix)

        self._intialize()
        self.animal = Animal(self)
        self.probemap = Probemap(self)

    def _intialize(self):

        self.sampfreq = None
        self.channels = None
        self.nChans = None
        self.lfpSrate = None
        self.channelgroups = None
        self.badchans = None
        self.nShanks = None
        self.auxchans = None
        self.skulleeg = None
        self.emgChans = None
        self.motionChans = None
        self.nShanksProbe = None
        self.nProbes = None

        if self.files.basics.is_file():
            myinfo = np.load(self.files.basics, allow_pickle=True).item()
            for attrib, val in myinfo.items():  # alternative list(epochs)
                setattr(self, attrib, val)  # .lower() will be removed

            self.goodchans = np.setdiff1d(
                self.channels, self.badchans, assume_unique=True
            )
            self.goodchangrp = [
                list(np.setdiff1d(_, self.badchans, assume_unique=True).astype(int))
                for _ in self.channelgroups
            ]

    def __str__(self) -> str:
        return f"Name: {self.session.name} \nChannels: {self.nChans}\nSampling Freq: {self.sampfreq}\nlfp Srate (downsampled): {self.lfpSrate}\n# bad channels: {len(self.badchans)}\nmotion channels: {self.motionChans}\nemg channels: {self.emgChans}\nskull eeg: {self.skulleeg}"

    def generate_xml(self, settingsPath):
        """Generates .xml for the data using openephys's settings.xml"""
        myroot = ET.parse(settingsPath).getroot()

        chanmap = []
        for elem in myroot[1][1][-1]:
            if "Mapping" in elem.attrib:
                chanmap.append(elem.attrib["Mapping"])

        neuroscope_xmltree = ET.parse(self.files.filePrefix.with_suffix(".xml"))
        neuroscope_xmlroot = neuroscope_xmltree.getroot()

        for i, chan in enumerate(neuroscope_xmlroot[2][0][0].iter("channel")):
            chan.text = str(int(chanmap[i]) - 1)

        neuroscope_xmltree.write(self.files.filePrefix.with_suffix(".xml"))

    def makerecinfo(self, nShanks=None, skulleeg=None, emg=None, motion=None):
        """Uses .xml file to parse anatomical groups

        Parameters
        ----------
        nShanks : int or list of int, optional
            number of shanks, if None then this equals to number of anatomical grps excluding channels mentioned in skulleeg, emg, motion
        skulleeg : list, optional
            any channels recorded from the skull, by default None
        emg : list, optional
            emg channels, by default None
        motion : list, optional
            channels recording accelerometer data or velocity, by default None
        """

        if skulleeg is None:
            skulleeg = []
        if emg is None:
            emg = []
        if motion is None:
            motion = []

        skulleeg = list(skulleeg)
        emg = list(emg)
        motion = list(motion)

        myroot = ET.parse(self.recfiles.xmlfile).getroot()

        chan_session, channelgroups, badchans = [], [], []
        for x in myroot.findall("anatomicalDescription"):
            for y in x.findall("channelGroups"):
                for z in y.findall("group"):
                    chan_group = []
                    for chan in z.findall("channel"):
                        if int(chan.text) not in skulleeg + emg + motion:
                            chan_session.append(int(chan.text))
                            if int(chan.attrib["skip"]) == 1:
                                badchans.append(int(chan.text))

                            chan_group.append(int(chan.text))
                    if chan_group:
                        channelgroups.append(chan_group)

        sampfreq = nChans = None
        for sf in myroot.findall("acquisitionSystem"):
            sampfreq = int(sf.find("samplingRate").text)
            nChans = int(sf.find("nChannels").text)

        lfpSrate = None
        for val in myroot.findall("fieldPotentials"):
            lfpSrate = int(val.find("lfpSamplingRate").text)

        auxchans = np.setdiff1d(
            np.arange(nChans), np.array(chan_session + skulleeg + emg + motion)
        )
        if auxchans.size == 0:
            auxchans = None

        if nShanks is None:
            nShanks = len(channelgroups)

        nShanksProbe = [nShanks] if isinstance(nShanks, int) else nShanks
        nProbes = len(nShanksProbe)
        nShanks = np.sum(nShanksProbe)

        if motion is not None:
            pass

        basics = {
            "sampfreq": sampfreq,
            "channels": chan_session,
            "nChans": nChans,
            "channelgroups": channelgroups,
            "nShanks": nShanks,
            "nProbes": nProbes,
            "nShanksProbe": nShanksProbe,
            "subname": self.session.subname,
            "sessionName": self.session.sessionName,
            "lfpSrate": lfpSrate,
            "badchans": badchans,
            "auxchans": auxchans,
            "skulleeg": skulleeg,
            "emgChans": emg,
            "motionChans": motion,
        }

        np.save(self.files.basics, basics)
        print(f"_basics.npy created for {self.session.sessionName}")

        # laods attributes in runtime so doesn't lead reloading of entire class instance
        self._intialize()

    @property
    def getNframesDat(self):
        nChans = self.nChans
        datfile = self.recfiles.datfile
        data = np.memmap(datfile, dtype="int16", mode="r")

        return len(data) / nChans

    @property
    def getNframesEEG(self):
        nframes = len(self.geteeg(chans=0))

        return nframes

    @property
    def duration(self):
        return self.getNframesEEG / self.lfpSrate

    def time_slice(self, chans, period=None):
        """Returns eeg signal for given channels. If multiple channels provided then it is list of lfps.

        Args:
            chans (list/array): channels required, index should in order of binary file
            timeRange (list, optional): In seconds and must have length 2.

        Returns:
            eeg: memmap, or list of memmaps
        """
        eegfile = self.recfiles.eegfile
        eegSrate = self.lfpSrate
        nChans = self.nChans

        if period is not None:
            assert len(period) == 2
            frameStart = int(period[0] * eegSrate)
            frameEnd = int(period[1] * eegSrate)
            eeg = np.memmap(
                eegfile,
                dtype="int16",
                mode="r",
                offset=2 * nChans * frameStart,
                shape=nChans * (frameEnd - frameStart),
            )
            eeg = np.memmap.reshape(eeg, (nChans, len(eeg) // nChans), order="F")

        else:
            eeg = np.memmap(eegfile, dtype="int16", mode="r")
            eeg = np.memmap.reshape(eeg, (nChans, len(eeg) // nChans), order="F")

        eeg_ = []
        if isinstance(chans, (list, np.ndarray)):
            for chan in chans:
                eeg_.append(eeg[chan, :])
        else:
            eeg_ = eeg[chans, :]

        return eeg_

    def geteeg(self, chans, timeRange=None):
        """Returns eeg signal for given channels. If multiple channels provided then it is list of lfps.

        Args:
            chans (list/array): channels required, index should in order of binary file
            timeRange (list, optional): In seconds and must have length 2.

        Returns:
            eeg: memmap, or list of memmaps
        """
        eegfile = self.recfiles.eegfile
        eegSrate = self.lfpSrate
        nChans = self.nChans

        if timeRange is not None:
            assert len(timeRange) == 2
            frameStart = int(timeRange[0] * eegSrate)
            frameEnd = int(timeRange[1] * eegSrate)
            eeg = np.memmap(
                eegfile,
                dtype="int16",
                mode="r",
                offset=2 * nChans * frameStart,
                shape=nChans * (frameEnd - frameStart),
            )
            eeg = np.memmap.reshape(eeg, (nChans, len(eeg) // nChans), order="F")

        else:
            eeg = np.memmap(eegfile, dtype="int16", mode="r")
            eeg = np.memmap.reshape(eeg, (nChans, len(eeg) // nChans), order="F")

        eeg_ = []
        if isinstance(chans, (list, np.ndarray)):
            for chan in chans:
                eeg_.append(eeg[chan, :])
        else:
            eeg_ = eeg[chans, :]

        return eeg_

    def getPxx(self, chan, window=2, overlap=1, period=None):
        """Get powerspectrum

        Parameters
        ----------
        chan : int
            channel for which to calculate the power spectrum
        period : list/array, optional
            lfp is restricted to only this period, by default None

        Returns
        -------
        [type]
            [description]
        """
        eeg = self.geteeg(chans=chan, timeRange=period)
        f, pxx = sg.welch(
            eeg,
            fs=self.lfpSrate,
            nperseg=int(window * self.lfpSrate),
            noverlap=int(overlap * self.lfpSrate),
        )
        return f, pxx

    def get_spectrogram(self, chan, window=5, overlap=2, period=None, **kwargs):
        """Get spectrogram for a given lfp channel (from .eeg file)

        Parameters
        ----------
        chan : int
            channel number
        window : float, optional
            size of the window, in seconds, by default 5
        overlap : float, optional
            overlap between adjacent window, in seconds, by default 2
        period : list, optional
            calculate spectrogram within this period only, list of length 2, in seconds, by default None

        Returns
        -------
        list of arrays
            frequency, time in seconds, spectrogram
        """

        lfp = self.geteeg(chans=chan, timeRange=period)
        f, t, sxx = sg.spectrogram(
            lfp,
            fs=self.lfpSrate,
            nperseg=window * self.lfpSrate,
            noverlap=overlap * self.lfpSrate,
            **kwargs,
        )

        return f, t, sxx

    def plot_spectrogram(
        self, chan=None, period=None, window=10, overlap=2, ax=None, plotChan=False
    ):
        """Generating spectrogram plot for given channel

        Parameters
        ----------
        chan : [int], optional
            channel to plot, by default None and chooses a channel randomly
        period : [type], optional
            plot only for this duration in the session, by default None
        window : [float, seconds], optional
            window binning size for spectrogram, by default 10
        overlap : [float, seconds], optional
            overlap between windows, by default 2
        ax : [obj], optional
            if none generates a new figure, by default None
        """

        if chan is None:
            goodchans = self._obj.goodchans
            chan = np.random.choice(goodchans)

        eegSrate = self._obj.lfpSrate
        lfp = self._obj.geteeg(chans=chan, timeRange=period)

        spec = signal_process.spectrogramBands(
            lfp, sampfreq=eegSrate, window=window, overlap=overlap
        )

        sxx = spec.sxx / np.max(spec.sxx)
        sxx = gaussian_filter(sxx, sigma=2)

        if ax is None:
            _, ax = plt.subplots(1, 1)

        ax = plotting.plot_spectrogram(sxx, ax=ax)
        ax.text(
            np.max(spec.time) / 2,
            25,
            f"Spectrogram for channel {chan}",
            ha="center",
            color="w",
        )
        ax.set_xlim([np.min(spec.time), np.max(spec.time)])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        if plotChan:
            axins = ax.inset_axes([0, 0.6, 0.1, 0.25])
            self._obj.probemap.plot(chans=[chan], ax=axins)
            axins.axis("off")

    def loadmetadata(self):
        metadatafile = Path(str(self.files.filePrefix) + "_metadata.csv")
        metadata = pd.read_csv(metadatafile)

        return metadata


class files:
    def __init__(self, filePrefix):
        self.filePrefix = filePrefix
        self.probe = filePrefix.with_suffix(".probe.npy")
        self.basics = Path(str(filePrefix) + "_basics.npy")
        self.position = Path(str(filePrefix) + "_position.npy")
        self.epochs = Path(str(filePrefix) + "_epochs.npy")
        self.spectrogram = Path(str(filePrefix) + "_sxx.npy")


class recfiles:
    def __init__(self, f_prefix):

        self.xmlfile = f_prefix.with_suffix(".xml")
        self.eegfile = f_prefix.with_suffix(".eeg")
        self.datfile = f_prefix.with_suffix(".dat")


class sessionname:
    def __init__(self, f_prefix):
        basePath = str(f_prefix.parent.as_posix())
        self.sessionName = basePath.split("/")[-2] + basePath.split("/")[-1]
        self.name = basePath.split("/")[-2]
        self.day = basePath.split("/")[-1]
        # self.basePath = Path(basePath)
        self.subname = f_prefix.stem
