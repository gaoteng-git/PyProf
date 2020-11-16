#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Parse the SQLite3 database from NVprof or Nsight and print a dictionary for every kernel.
"""

import sys
import os
import argparse
from tqdm import tqdm

from .db import DB
from .kernel import Kernel
from .nvvp import NVVP
from .nsight import Nsight

# ===================== Overall View ========================
# Need to "pip install python-intervals"
# import intervals as I
import json

class Section:
    def __init__(self):
        self.name = None
        self.startTime = None
        self.endTime = None
        self.durationTime = None # Infered by "self.endTime - self.startTime"

    def buildNode(self, r):
        self.name = r['name']
        self.startTime = int(r['start'])
        if r['end']: # TODO: Remove it. Sometimes not using "-c cudaProfilerApi --stop-on-range-end true" will lead to no "end".
            self.endTime = int(r['end'])
        else:
            self.endTime = sys.maxsize        
        self.durationTime = self.endTime - self.startTime 


class RuntimeCall(Section):
    def __init__(self):
        Section.__init__(self)
        self.corralationId = None    


class KernelExecute(Section):
    def __init__(self):
        Section.__init__(self)
        self.corralationId = None    
        self.streamId = None

def getStepList(db):
    steps = []
    cmd = "SELECT start, end, text as name, globalTid FROM NVTX_EVENTS \
        WHERE name LIKE 'iter-%'\
        ORDER BY start ASC"
    result = db.select(cmd)    
    for r in result:
        s = Section()
        s.buildNode(r)
        steps.append(s)
    return steps


# Each thread has a list.
def getRuntimeLists(db):
    runtimeLists = {}
    cmd = 'SELECT value AS name, start, end, globalTid, correlationId \
        FROM CUPTI_ACTIVITY_KIND_RUNTIME, StringIds \
        WHERE CUPTI_ACTIVITY_KIND_RUNTIME.NameId = StringIds.id \
        ORDER BY start ASC'
    result = db.select(cmd)    
    for r in result:
        globalTid = r['globalTid']
        tid = globalTid & 0x00000000ffffff  # not sure, but appears to be.
        if not runtimeLists.__contains__(tid):
            runtimeLists[tid] = []
        rtc = RuntimeCall()
        rtc.buildNode(r)
        rtc.corralationId = r['correlationId']
        runtimeLists[tid].append(rtc)        
    return runtimeLists


# Each stream has a list.
# TODO: deviceId, globalPid
def getKernelExecuteLists(db):
    kernelExecLists = {}
    cmd = 'SELECT StringIds.value AS name, start, end, streamId, correlationId \
        FROM CUPTI_ACTIVITY_KIND_KERNEL, StringIds \
        WHERE CUPTI_ACTIVITY_KIND_KERNEL.shortName = StringIds.id \
        ORDER BY start ASC'
    result = db.select(cmd)    
    for r in result:
        sid = r['streamId']
        if not kernelExecLists.__contains__(sid):
            kernelExecLists[sid] = []
        ke = KernelExecute()
        ke.buildNode(r)
        ke.corralationId = r['correlationId']
        ke.streamId = r['streamId']
        kernelExecLists[sid].append(ke)
    return kernelExecLists


def getCudaMemcpyLists(db):
    cudaMemcpyLists = {}
    cmd = 'SELECT copyKind AS name, start, end, streamId, correlationId \
        FROM CUPTI_ACTIVITY_KIND_MEMCPY \
        ORDER BY start ASC'
    result = db.select(cmd)    
    for r in result:
        sid = r['streamId']
        if not cudaMemcpyLists.__contains__(sid):
            cudaMemcpyLists[sid] = []
        ke = KernelExecute() # Reuse.
        ke.buildNode(r)
        ke.corralationId = r['correlationId']
        ke.streamId = r['streamId']
        cudaMemcpyLists[sid].append(ke)
    return cudaMemcpyLists


# NVTX_EVENTS contains "traceMarker" and "dataloader.py"
def getDataLoaderLists(db):
    dataLoaderLists = {}
    cmd = "SELECT text AS name, start, end, globalTid FROM NVTX_EVENTS \
        WHERE text LIKE '%''name'': ''torch.utils.data.dataloader.DataLoader''%' ORDER BY start ASC"
    result = db.select(cmd)    
    for r in result:
        globalTid = r['globalTid']
        tid = globalTid & 0x00000000ffffff  # not sure, but appears to be. # TODO: Should we use PID???!!!
        if not dataLoaderLists.__contains__(tid):
            dataLoaderLists[tid] = []
        rtc = Section()
        rtc.buildNode(r)
        dataLoaderLists[tid].append(rtc)        
    return dataLoaderLists


# NVTX_EVENTS except "iter-XX"
def getHostComputeLists(db):
    hostComputeLists = {}
    cmd = "SELECT text AS name, start, end, globalTid FROM NVTX_EVENTS WHERE text NOT LIKE 'iter-%' ORDER BY start ASC"
    result = db.select(cmd)    
    for r in result:
        globalTid = r['globalTid']
        tid = globalTid & 0x00000000ffffff  # not sure, but appears to be.
        if not hostComputeLists.__contains__(tid):
            hostComputeLists[tid] = []
        rtc = Section()
        rtc.buildNode(r)
        hostComputeLists[tid].append(rtc)        
    return hostComputeLists


# <tid, [runtimeCall]> --> [step: <tid, [runtimeCall]>]
def splitRuntimeListsByStep(runtimeLists, stepList):
    runtimeListsByStep = []
    for step in stepList:
        runtimeListsThisStep = {}
        for (t, rtcList) in runtimeLists.items():
            for rtc in rtcList:
                if rtc.startTime >= step.startTime and rtc.endTime <= step.endTime:
                    if not runtimeListsThisStep.__contains__(t):
                        runtimeListsThisStep[t] = []
                    runtimeListsThisStep[t].append(rtc)
        runtimeListsByStep.append(runtimeListsThisStep)
    return runtimeListsByStep


# <sid, [kernelExec]> --> [step: <sid, [kernelExec]>]
def splitKernelExecuteListsByStep(kernelExecLists, runtimeListsByStep):
    kernelExecListsByStep = []
    for i in range(len(runtimeListsByStep)):
        kernelExecListsThisStep = {}
        runtimeListsThisStep = runtimeListsByStep[i]
        if runtimeListsThisStep:
            corrIdSet = set()
            for (t, rtList) in runtimeListsThisStep.items():
                for rt in rtList:
                    corrIdSet.add(rt.corralationId)
            for (s, kernelExecList) in kernelExecLists.items():
                for ke in kernelExecList:
                    if ke.corralationId in corrIdSet:
                        if not kernelExecListsThisStep.__contains__(s):
                            kernelExecListsThisStep[s] = []
                        kernelExecListsThisStep[s].append(ke)
        kernelExecListsByStep.append(kernelExecListsThisStep)
    return kernelExecListsByStep

def MergeRuntimeCallToInterval(runtimeLists):
    #TODO: considier multi-threads overlap.
    resultInterval = I.empty()
    for (t, rtcList) in runtimeLists.items():
        rtcInterval = SectionListToInterval(rtcList)
        resultInterval |= rtcInterval
    return resultInterval

def MergeKernelExecToInterval(kernelExecLists):
    resultInterval = I.empty()
    for (s, kernelExecList) in kernelExecLists.items():
        subInterval = SectionListToInterval(kernelExecList)
        resultInterval = resultInterval | subInterval
    return resultInterval


def ConvertIntervalToClosed(interval):
    return interval.apply(lambda x: (I.CLOSED, x.lower, x.upper, I.CLOSED))

def GetInervalSum(interval):
    intervalPython = I.to_data(interval)
    sum = 0
    for intervalTuple in intervalPython:
        duration = intervalTuple[2] - intervalTuple[1]
        sum += duration
    return sum

def SectionToInterval(section):
    return I.closed(section.startTime, section.endTime)

def SectionListToInterval(sectionList):
    resultInterval = I.empty()
    for l in sectionList:
        interval = SectionToInterval(l)
        # TODO: Check if result overlap with interval
        resultInterval = resultInterval | interval
    return resultInterval


def GetKernelSlots(step, kernelExecInterval):
    stepInterval = SectionToInterval(step)    
    
    slots = ConvertIntervalToClosed(slots)
    return slots

def GetRuntimeSlots(kernelSlots, runtimeCallInterval):
    runtimeSlots = kernelSlots - runtimeCallInterval
    runtimeSlots = ConvertIntervalToClosed(runtimeSlots)
    return runtimeSlots

def Run(stepList, runtimeListsByStep, kernelExecListsByStep, cudaMemcpyListsByStep, dataLoaderListsByStep, hostExecListsByStep):
    overallDict = {}
    overallDict["X_lables"] = []
    overallDict["iterations"] = {"KernelExec":[], "CudaMemcpy":[], "CudaRuntime":[], "DataLoader":[], "HostExec":[], "Other":[]}
    for iStep in range(1, len(stepList)): # Skip 1st step.
        overallDict["X_lables"].append(stepList[iStep].name)

        kernelExecInterval = MergeKernelExecToInterval(kernelExecListsByStep[iStep])

        cudaMemcpyIntervalAll = MergeKernelExecToInterval(cudaMemcpyListsByStep[iStep]) # Reuse.
        cudaMemcpyIntervalPure = cudaMemcpyIntervalAll - kernelExecInterval

        runtimeCallIntervalAll = MergeRuntimeCallToInterval(runtimeListsByStep[iStep])
        runtimeCallIntervalPure = runtimeCallIntervalAll - kernelExecInterval - cudaMemcpyIntervalAll

        dataLoaderIntervalAll = MergeRuntimeCallToInterval(dataLoaderListsByStep[iStep])
        dataLoaderIntervalPure = dataLoaderIntervalAll - kernelExecInterval - cudaMemcpyIntervalAll - runtimeCallIntervalAll

        hostExecIntervalAll = MergeRuntimeCallToInterval(hostExecListsByStep[iStep]) # Reuse.        
        hostExecIntervalPure = hostExecIntervalAll - dataLoaderIntervalAll - kernelExecInterval - cudaMemcpyIntervalAll - runtimeCallIntervalAll

        stepInterval = SectionToInterval(stepList[iStep])    

        if not kernelExecInterval.is_empty() and kernelExecInterval.upper > stepInterval.upper:
            print("WARNING: kernelExecInterval.upper > stepInterval.upper")        
        otherInterval = stepInterval - kernelExecInterval - cudaMemcpyIntervalAll - runtimeCallIntervalAll - hostExecIntervalAll

        kernelExecIntervalSum = GetInervalSum(kernelExecInterval)
        overallDict["iterations"]["KernelExec"].append(kernelExecIntervalSum)
        cudaMemcpyIntervalPureSum = GetInervalSum(cudaMemcpyIntervalPure)
        overallDict["iterations"]["CudaMemcpy"].append(cudaMemcpyIntervalPureSum)
        runtimeCallIntervalPureSum = GetInervalSum(runtimeCallIntervalPure)
        overallDict["iterations"]["CudaRuntime"].append(runtimeCallIntervalPureSum)
        dataLoaderIntervalPureSum = GetInervalSum(dataLoaderIntervalPure)
        overallDict["iterations"]["DataLoader"].append(dataLoaderIntervalPureSum)
        hostExecIntervalPureSum = GetInervalSum(hostExecIntervalPure)
        overallDict["iterations"]["HostExec"].append(hostExecIntervalPureSum)
        otherIntervalSum = GetInervalSum(otherInterval)
        overallDict["iterations"]["Other"].append(otherIntervalSum)
    return overallDict


# ===================== Operator View & Kernel View ========================
def parseArgs():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Parse SQLite3 DB from NVprof or Nsight.")
    parser.add_argument("file", type=str, default=None, help="SQLite3 database.")

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise parser.error("No such file '{}'.".format(args.file))

    return args

class Agg:
    def __init__(self):
        self.durationTime = 0
        self.kDurationTime = 0
        self.mDurationTime = 0

class KAgg:
    def __init__(self):
        self.duration = 0
        self.calls = 0

class TimeRange:
    def __init__(self):
        self.startTime = 0
        self.endTime = 0

def ExtractName(fullname):
        parts = fullname.split(' | ')
        for part in parts:
            d = eval(part)
            if d.__contains__("name"):
                return d['name']
        return "None"

def ProcessName(name):
        if name.__contains__(" | "):
            name = ExtractName(name)
        elif name.__contains__(", seq = "):
            pos_index = name.find(", seq = ")
            name = name[:pos_index]
        return name

class TreeNode:
    def __init__(self):
        self.name = None
        self.startTime = None
        self.endTime = None
        self.father = None
        self.children = []
        self.durationTime = None # Infered by "self.endTime - self.startTime"

        self.rtDurationTime = 0 # Runtime Kernel Launch Duration Time
        self.kDurationTime = 0 # Kernel Execute Duration Time
        self.mDurationTime = 0 # GPU Memcpy & Memset Time
        self.sDurationTime = 0 # GPU Sync Time

        self.startTimeDevice = None
        self.endTimeDevice = None

    def buildNode(self, r):
        self.name = r['name']
        self.startTime = int(r['start'])
        self.endTime = int(r['end'])
        self.startTime /= 1000 # Unit from ns to us.
        self.endTime /= 1000
        self.durationTime = self.endTime - self.startTime 

    def insertChildren(self, node):
        if node.endTime < self.startTime or node.startTime > self.endTime:
            print("ERROR: Fail to insert children!")
            return
        isSucc = True
        childrenNum = len(self.children)
        if childrenNum == 0 or node.startTime > self.children[-1].endTime:
            self.children.append(node)
            node.father = self
            return isSucc
        for index in range(childrenNum):
            child = self.children[index]
            if node.endTime < child.startTime:
                self.children.insert(index, node)
                node.father = self
                break
            if node.startTime < child.startTime:
                print("ERROR: Fail to insert children! Intersection!")                
                isSucc = False
                break
            if node.endTime <= child.endTime:
                isSucc = child.insertChildren(node)
                break
            if node.startTime <= child.endTime:
                print("ERROR: Fail to insert children! Intersection!")                
                isSucc = False
                break        
        return isSucc    

    def print(self, level):
        PRINT_SPACES = 4
        spaces = ' ' * (level * PRINT_SPACES)        
        line = self.toString(spaces)
        print(line)
        for child in self.children:
            child.print(level + 1)
    
    def chrome_trace(self, tid, file):
        str = self.toChromeString(tid)
        file.write(str)
        for child in self.children:
            child.chrome_trace(tid, file)
    
    def chrome_trace_device(self, file):
        str = self.toChromeStringDevice()
        if str != "":
            file.write(str)
        for child in self.children:
            child.chrome_trace_device(file)
    
    def chrome_trace_device_merged(self, file, mergedTime):
        childMergedTime = mergedTime
        str, mergedTime = self.toChromeStringDeviceMerged(mergedTime)
        if str != "":
            file.write(str)
        for child in self.children:
            childMergedTime = child.chrome_trace_device_merged(file, childMergedTime)        
        return mergedTime
    
    def aggregate_op(self, iter, result):            
        if not "Root" in self.name \
            and not "iter-" in self.name \
            and not ("{" in self.name and "}" in self.name):
            return
        isOperator = True
        if "iter-" in self.name or "Root" in self.name:
            iter = self.name
            isOperator = False
        else:
            for child in self.children:
                if "{" in child.name and "}" in child.name:
                    isOperator = False
                    break
        if "Tensor.backward" in self.name: # TODO: it has sub-operator in it! But all its sub-operators are C++
            isOperator = True
        if isOperator and iter:
            if not result.__contains__(iter):
                result[iter] = {}
            name = ExtractName(self.name)
            if not result[iter].__contains__(name):
                result[iter][name] = Agg()
            result[iter][name].durationTime += self.durationTime
            result[iter][name].kDurationTime += self.kDurationTime
            result[iter][name].mDurationTime += self.mDurationTime
        else:
            for child in self.children:
                child.aggregate_op(iter, result)

    def aggregate_kernel(self, isMainThread, minEndTime, op, kernelResult, combineResult):
        if not isMainThread:
            op = "Tensor.backward" # TODO: it has sub-operator in it! But all its sub-operators are C++
        if self.endTime <= minEndTime: # Filter the 1st iteration.
            return
        name = str(self.name)
        if name != "Root" and (not "iter-" in name) and ("{" in name and "}" in name):
            name = ExtractName(self.name)            
            op = name # The lowest {'name':'XXX'} will be regarded as the operator containing kernels info.
        for child in self.children:
            child.aggregate_kernel(isMainThread, minEndTime, op, kernelResult, combineResult)

    def toString(self, spaces):
        nodeStr = spaces + "<CPU-Elapsed:" + str(self.durationTime) + ",  " + "<LaunchKernel:" + str(self.rtDurationTime) + ">>" + "  Device:<" + "Kernel:" + str(self.kDurationTime) + ", " + "Memcpy:" + str(self.mDurationTime) + ">" + "  " + "Name:(" + str(self.name) + ")  " + "StartTime:" + str(self.startTime) + "  " + "EndTime:" + str(self.endTime)
        return nodeStr

    def toChromeString(self, tid):
        name = ProcessName(self.name)        
        chrome_str = "{\"name\": \"" + str(name) + "\", " \
                + "\"ph\": \"X\", " \
                + "\"ts\": " + str(self.startTime / 1000) + ", " \
                + "\"dur\": " + str(self.durationTime / 1000) + ", " \
                + "\"tid\": " + str(tid) + ", " \
                + "\"pid\": \"CPU functions\", " \
                + "\"args\": {}}, "
        return chrome_str

    def toChromeStringDevice(self):
        if not self.startTimeDevice:
            return ""
        name = ProcessName(self.name)
        d = self.endTimeDevice - self.startTimeDevice
        chrome_str = "{\"name\": \"" + str(name) + "\", " \
                + "\"ph\": \"X\", " \
                + "\"ts\": " + str(self.startTimeDevice / 1000) + ", " \
                + "\"dur\": " + str((d) / 1000) + ", " \
                + "\"pid\": \"Device functions\", " \
                + "\"args\": {}}, "
        return chrome_str

    def toChromeStringDeviceMerged(self, mergedStartTime):
        if not self.startTimeDevice:
            return "", mergedStartTime
        name = ProcessName(self.name)
        duration = self.kDurationTime + self.mDurationTime # TODO: Deal with multi-stream overlap!!
        if duration == 0:
            return "", mergedStartTime
        chrome_str = "{\"name\": \"" + str(name) + "\", " \
                + "\"ph\": \"X\", " \
                + "\"ts\": " + str(mergedStartTime / 1000) + ", " \
                + "\"dur\": " + str(duration / 1000) + ", " \
                + "\"pid\": \"Device functions\", " \
                + "\"args\": {}}, "
        return chrome_str, mergedStartTime + duration

        
    def toString_PPT(self):
        originName = str(self.name)
        dstName = originName
        if "{" in originName and "}" in originName:
            parts = originName.split(' | ')
            for part in parts:
                d = eval(part)
                if d.__contains__("name"):
                    dstName = d['name']
        if ", seq = " in originName:
            dstName = originName.split(',')[0]
        divide = 1000 * 1000
        nodeStr = dstName + "\t" + str(round(self.durationTime / divide, 2)) + "\t" + str(round(self.kDurationTime / divide, 2))
        return nodeStr

    def Summarize(self):
        self.rtDurationTime = 0
        self.kDurationTime = 0
        self.mDurationTime = 0
        self.sDurationTime = 0
        self.startTimeDevice = None
        self.endTimeDevice = None
        for child in self.children:
            child.Summarize()
            self.rtDurationTime += child.rtDurationTime
            self.kDurationTime += child.kDurationTime            
            self.mDurationTime += child.mDurationTime
            self.sDurationTime += child.sDurationTime
            if child.startTimeDevice:
                if not self.startTimeDevice:
                    self.startTimeDevice = child.startTimeDevice
                    self.endTimeDevice = child.endTimeDevice
                else:
                    self.startTimeDevice = min(self.startTimeDevice, child.startTimeDevice)
                    self.endTimeDevice = max(self.endTimeDevice, child.endTimeDevice)



class TreeNodeRuntime(TreeNode):
    def __init__(self):
        TreeNode.__init__(self)
        self.cudaNode = None     

    def toString(self, spaces):
        nodeStr = spaces + "<CPU-Elapsed:" + str(self.durationTime) + ",  " + "<LaunchKernel:" + str(self.rtDurationTime) + ">>" + "  Device:<" + "Kernel:" + str(self.kDurationTime) + ", " + "Memcpy:" + str(self.mDurationTime) + ">" + "  " + "Name:[" + str(self.name) + "]  " + "StartTime:" + str(self.startTime) + "  " + "EndTime:" + str(self.endTime)
        if self.cudaNode:
            nodeStr += "\n"        
            nodeStr += self.cudaNode.toString(spaces)
        return nodeStr

    def toChromeString(self, tid):
        chrome_str = TreeNode.toChromeString(self, tid)
        if self.cudaNode:
            chrome_str += self.cudaNode.toChromeString()
        return chrome_str

    def Summarize(self):
        self.rtDurationTime = self.durationTime
        self.kDurationTime = 0
        self.mDurationTime = 0
        self.sDurationTime = 0
        self.startTimeDevice = None
        self.endTimeDevice = None
        if self.cudaNode:
            self.cudaNode.Summarize()
            self.kDurationTime = self.cudaNode.kDurationTime
            self.mDurationTime = self.cudaNode.mDurationTime
            self.sDurationTime = self.cudaNode.sDurationTime
            self.startTimeDevice = self.cudaNode.startTimeDevice
            self.endTimeDevice = self.cudaNode.endTimeDevice
    
    def aggregate_kernel(self, isMainThread, minEndTime, op, kernelResult, combineResult):
        if not isMainThread:
            op = "Tensor.backward"
        if self.endTime <= minEndTime: # Filter the 1st iteration.
            return
        if self.cudaNode:
            self.cudaNode.aggregate_kernel(isMainThread, minEndTime, op, kernelResult, combineResult)

class TreeNodeCUDA(TreeNode):
    def __init__(self):
        TreeNode.__init__(self)
        self.streamId = None

    def buildNode(self, r):
        TreeNode.buildNode(self, r)
        self.streamId = r["streamId"]
        self.deviceId = r['deviceId']
        self.contextId = r['contextId']        

    def Summarize(self):        
        self.startTimeDevice = self.startTime
        self.endTimeDevice = self.endTime

    def toString(self, spaces):
        nodeStr = spaces + "    " + "<CPU-Elapsed:" + str(self.durationTime) + ",  " + "<LaunchKernel:" + str(self.rtDurationTime) + ">>" + "  Device:<" + "Kernel:" + str(self.kDurationTime) + ", " + "Memcpy:" + str(self.mDurationTime) + ">" + "  " + "Name:{" + str(self.name) + "}  " + "StartTime:" + str(self.startTime) + "  " + "EndTime:" + str(self.endTime)
        return nodeStr

    def toChromeString(self):
        chrome_str = "{\"name\": \"" + str(self.name) + "\", " \
                + "\"ph\": \"X\", " \
                + "\"ts\": " + str(self.startTime / 1000) + ", " \
                + "\"dur\": " + str(max(self.kDurationTime, self.mDurationTime) / 1000) + ", " \
                + "\"tid\": " + str(self.streamId) + ", " \
                + "\"pid\": \"CUDA functions\", " \
                + "\"args\": {}}, "
        return chrome_str

class TreeNodeKernel(TreeNodeCUDA):
    def Summarize(self):
        TreeNodeCUDA.Summarize(self)
        self.kDurationTime = self.durationTime        
    
    def aggregate_kernel(self, isMainThread, minEndTime, op, kernelResult, combineResult):        
        if not kernelResult.__contains__(self.name):
            kernelResult[self.name] = KAgg()
        kernelResult[self.name].duration += self.kDurationTime
        kernelResult[self.name].calls += 1
        combineName = op + "|" + self.name
        if not combineResult.__contains__(combineName):
            combineResult[combineName] = KAgg()
        combineResult[combineName].duration += self.kDurationTime
        combineResult[combineName].calls += 1


class TreeNodeMemcpy(TreeNodeCUDA):
    def buildNode(self, r):
        TreeNodeCUDA.buildNode(self, r)        
        # copyKind meaning got from "enum CUpti_ActivityMemcpyKind"
        if str(self.name) == "1":
            self.name = "Memcpy HtoD (Host -> Device)" 
        elif str(self.name) == "2":
            self.name = "Memcpy DtoH (Device -> Host)" 
        elif str(self.name) == "8":
            self.name = "Memcpy DtoD (Device -> Device)" 
        else:
            self.name = "Memcpy OtherKind"
        self.bytes = r["bytes"]
        self.bandwidth = 1000000 / (1024 * 1024 * 1024) * self.bytes / self.durationTime

    def Summarize(self):
        TreeNodeCUDA.Summarize(self)
        self.mDurationTime = self.durationTime


class TreeNodeMemset(TreeNodeCUDA):
    def buildNode(self, r):
        TreeNodeCUDA.buildNode(self, r)
        self.bytes = r["bytes"]
        self.bandwidth = 1000000 / (1024 * 1024 * 1024) * self.bytes / self.durationTime

    def Summarize(self):
        TreeNodeCUDA.Summarize(self)
        self.mDurationTime = self.durationTime

class TreeNodeSync(TreeNodeCUDA):
    def buildNode(self, r):
        TreeNodeCUDA.buildNode(self, r)
        self.name = "Sync"

    def Summarize(self):
        TreeNodeCUDA.Summarize(self)
        self.sDurationTime = self.durationTime


def buildNVTXTree(db, roots):   
    cmd = 'SELECT start, text FROM NVTX_EVENTS \
        WHERE text=\'__start_profile\''
    result = db.select(cmd) 
    start_time = None
    for r in result: 
        start_time = r['start']        
    cmd = 'SELECT start, end, text as name, globalTid FROM NVTX_EVENTS \
        WHERE start > ' + str(start_time) + ' ORDER BY start ASC'
    result = db.select(cmd)    
    for r in result:        
        if r['end'] == None:
            print("Skip: " + str(r))
            continue # TODO: endTime should be max?        
        globalTid = r['globalTid']
        #tid = globalTid & 0x00000000ffffff  # not sure, but appears to be.
        if not roots.__contains__(globalTid):
            root = TreeNode()
            root.startTime = 0
            root.endTime = sys.maxsize
            roots[globalTid] = root
        root = roots[globalTid]

        node = TreeNode()
        node.buildNode(r)
        node.durationTime = node.endTime - node.startTime

        if node.name.startswith("profiler::_record_function_"):
            continue # Filter it.
        root.insertChildren(node)
    print("buildNVTXTree finished.")


def buildRuntimeTree(db, roots, cudaDict):
    cmd = 'SELECT value AS name, start, end, globalTid, correlationId \
        FROM CUPTI_ACTIVITY_KIND_RUNTIME, StringIds \
        WHERE CUPTI_ACTIVITY_KIND_RUNTIME.NameId = StringIds.id'
    result = db.select(cmd)    
    for r in result:
        globalTid = r['globalTid']
        #tid = globalTid & 0x00000000ffffff  # not sure, but appears to be.
        if not roots.__contains__(globalTid):
            root = TreeNode()
            root.startTime = 0
            root.endTime = sys.maxsize
            roots[globalTid] = root
        root = roots[globalTid]

        node = TreeNodeRuntime()
        node.buildNode(r)        
        corralationId = r['correlationId'] # int
        if cudaDict.__contains__(corralationId):            
            cudaNode = cudaDict[corralationId]
            node.cudaNode = cudaNode

        root.insertChildren(node)
    print("buildRuntimeTree finished.")


def buildCUDADict(db):
    cudaDict = {}
    print("Before CUPTI_ACTIVITY_KIND_KERNEL")
    cmd = 'SELECT value AS name, start, end, correlationId, streamId, deviceId, contextId \
        FROM CUPTI_ACTIVITY_KIND_KERNEL, StringIds \
        WHERE CUPTI_ACTIVITY_KIND_KERNEL.shortName = StringIds.id'    
    try:
        result = db.select(cmd)        
        print("After CUPTI_ACTIVITY_KIND_KERNEL")
        for r in result:
            corralationId = r['correlationId']
            node = TreeNodeKernel()
            node.buildNode(r)
            cudaDict[corralationId] = node
    except:
        print("Exception when SELECT on CUPTI_ACTIVITY_KIND_KERNEL!")

    cmd = 'SELECT copyKind AS name, start, end, correlationId, streamId, deviceId, contextId, bytes \
        FROM CUPTI_ACTIVITY_KIND_MEMCPY'    
    try:
        result = db.select(cmd)
        for r in result:
            corralationId = r['correlationId']
            node = TreeNodeMemcpy()
            node.buildNode(r)
            cudaDict[corralationId] = node 
    except:  
        print("Exception when SELECT on CUPTI_ACTIVITY_KIND_MEMCPY!")

    cmd = 'SELECT start, end, correlationId, streamId, deviceId, contextId, bytes \
        FROM CUPTI_ACTIVITY_KIND_MEMSET'    
    try:
        result = db.select(cmd)
        for r in result:
            corralationId = r['correlationId']
            r["name"] = "Memset (Device)"
            node = TreeNodeMemset()
            node.buildNode(r)
            cudaDict[corralationId] = node
    except:  
        print("Exception when SELECT on CUPTI_ACTIVITY_KIND_MEMSET!")   

    cmd = 'SELECT syncType AS name, start, end, correlationId, streamId, deviceId, contextId \
        FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'    
    try:
        result = db.select(cmd)
        for r in result:
            corralationId = r['correlationId']
            node = TreeNodeSync()
            node.buildNode(r)
            cudaDict[corralationId] = node
    except:  
        print("Exception when SELECT on CUPTI_ACTIVITY_KIND_SYNCHRONIZATION!")    

    print("buildCUDADict finished.")
    return cudaDict


def RankDict(inDict):
    rankedList = []    
    for k, v in inDict.items():
        rankedList.append((k, v))
    rankedList.sort(key=lambda elem : elem[1].duration, reverse=True)
    return rankedList

def GetAllMainBackwards(mainNode, mainBackwards):
    if "Tensor.backward" in mainNode.name:
        mainBackwards.append(mainNode)
        return
    for child in mainNode.children:
        GetAllMainBackwards(child, mainBackwards)
    return

def SetBackwardThreadOpAsMainChildren(backwardNode, mainBackwards):
    for mainBackward in mainBackwards:
        if backwardNode.startTime >= mainBackward.startTime and backwardNode.endTime <= mainBackward.endTime:
            mainBackward.children.append(backwardNode)
            backwardNode.father = mainBackward            
            return
    isIntersect = False
    for mainBackward in mainBackwards:
        if not (backwardNode.endTime < mainBackward.startTime or backwardNode.startTime > mainBackward.endTime):
            isIntersect = True
            break    
    if isIntersect:
        for child in backwardNode.children:
            SetBackwardThreadOpAsMainChildren(child, mainBackwards)
    return

# ===================== chrome ======================
def export_chrome_trace(path, roots):
    vars = CommonVars()
    import os
    with open(path, 'w') as f:
        f.write('{' + '\n')
        f.write('"profilerMetadata": {' + '\n')
        f.write('    ' + '"DataSchemaVersion": "0.1.0",' + '\n')
        f.write('    ' + '"PyTorchVersion": "1.7.0"' + '\n')
        # TODO: DataLoader num_workers, pin_memory, ...
        f.write('},' + '\n')
        
        f.write('"traceEvents": [' + '\n')
        for globalTid, root in roots.items():
            adjust_step_device_time(root)
            tid = (globalTid & 0x00000000ffffff)  # not sure, but appears to be.
            pid = (globalTid >> 24) # TODO: This may be wrong!
            WriteTreeToChrome(root, tid, pid, vars, f)

        f.seek(f.tell() - 2, os.SEEK_SET) # Remove the last ',\n'
        f.write(']' + '\n')
        f.write('}' + '\n')

def adjust_step_device_time(root):
    def build_step(node, step_list):
        if (node.name.startswith("Python~~~Step~~~")):
            step_list.append(node)
        else:
            for child in node.children:
                build_step(child, step_list)
    step_list = []
    for node in root.children:
        build_step(node, step_list)
    if len(step_list) > 0 and (not step_list[0].startTimeDevice is None):
        for i in range(len(step_list)):
            if i == 0:
                # TODO: If there is warm-up steps before, then it may be not accurate!
                step_list[0].startTimeDevice = step_list[0].startTime            
            else:
                step_list[i].startTimeDevice = step_list[i - 1].endTimeDevice
    else:
        print("len(step_list) = " + str(len(step_list))) # Maybe backward, so no step.

def WriteTreeToChrome(node, tid, pid, vars, dstFile):
    if node.name != "Root":
        WriteNodeToChrome(node, tid, pid, vars, dstFile)
    for child in node.children:
        WriteTreeToChrome(child, tid, pid, vars, dstFile)

def WriteNodeToChrome(node, tid, pid, vars, dstFile):
    #print("WriteNodeToChrome: node.name=" + node.name)

    cpuRange = {}
    gpuRange = {}   
    cpuRange["args"] = gpuRange["args"] = {}
    cpuRange["ph"] = gpuRange["ph"] = "X"    
    cpuRange["ts"] = node.startTime # TODO: Unit???
    cpuRange["dur"] = node.durationTime # TODO: Unit???
    cpuRange["pid"] = gpuRange["pid"] = pid
    cpuRange["tid"] = tid
    cpuRange["args"]["Device"] = -1    
    corrId = vars.correlationId
    vars.correlationId += 1
    cpuRange["args"]["correlation"] = gpuRange["args"]["correlation"] = corrId
    cpuRange["args"]["stack"] = "stack_trace" # TODO: Real one
    if not node.startTimeDevice is None: # Should not show gpuRange if this is None!
        gpuRange["ts"] = node.startTimeDevice
        gpuRange["dur"] = node.endTimeDevice - node.startTimeDevice
        gpuRange["tid"] = "Device" # TODO: Is it good?        
    gpuRange["args"]["Device"] = 0 # TODO: Real one
    gpuRange["args"]["context"] = 1 # TODO: Real one
    gpuRange["args"]["stream"] = 7 # TODO: Real one

    if isinstance(node, TreeNodeRuntime):
        cat = "Runtime"
        cpuRange["cat"] = cat
        cpuRange["name"] = node.name
        cudaNode = node.cudaNode
        if (cudaNode is None) or (node.startTimeDevice is None):
            gpuRange = None
        else:
            gpuRange["tid"] = "stream " + str(cudaNode.streamId)
            gpuRange["name"] = cudaNode.name
            if (cudaNode.name is None) or (cudaNode.name == ""):
                print("Empty cudaNode.name!")
                return
            gpuRange["args"]["device"] = cudaNode.deviceId
            gpuRange["args"]["context"] = cudaNode.contextId
            gpuRange["args"]["stream"] = cudaNode.streamId
            if isinstance(cudaNode, TreeNodeKernel):                
                gpuRange["cat"] = "Kernel" 
            elif isinstance(cudaNode, TreeNodeMemset):
                gpuRange["cat"] = "Memset" 
                gpuRange["args"]["bytes"] = cudaNode.bytes
                gpuRange["args"]["memory bandwidth (GB/s)"] = cudaNode.bandwidth
            elif isinstance(cudaNode, TreeNodeMemcpy):
                gpuRange["cat"] = "Memcpy"
                gpuRange["args"]["bytes"] = cudaNode.bytes
                gpuRange["args"]["memory bandwidth (GB/s)"] = cudaNode.bandwidth
            else:
                gpuRange = None # TODO: Synchronize is not handled yet.
                return                           
    else:
        '''
        parts = node.name.split("###")
        if len(parts) != 2:
            print("parts!=2:  " + node.name)
            return
        fullName = parts[0]
        eventId = parts[1]
        '''
        fullName = node.name

        if fullName.startswith("Python~~~"):
            cat = "Python"
            cpuRange["cat"] = gpuRange["cat"] = cat
            parts = fullName.split("~~~")
            kind = parts[1]
            name = parts[2]
            if kind == "Step":
                cpuRange["name"] = gpuRange["name"] = "train_step"  
                stepId = int(name.split('-')[1])
                cpuRange["step"] = stepId
            elif kind == "DataLoader":
                cpuRange["name"] = gpuRange["name"] = "torch.utils.data.dataloader.DataLoader.__iter__"
            elif kind == "Forward":
                cpuRange["name"] = gpuRange["name"] = name
                cpuRange["Input dims"] = [] # TODO: Real one
                cpuRange["Input type"] = [] # TODO: Real one
                cpuRange["Input names"] = [] # TODO: Real one
            elif kind == "MemberFunc" and name == "Tensor.backward":
                cpuRange["name"] = gpuRange["name"] = "torch.Tensor.backward"
            elif kind == "MemberFunc" and name != "Tensor.backward":
                cpuRange["name"] = gpuRange["name"] = name
                cpuRange["Input dims"] = [] # TODO: Real one
                cpuRange["Input type"] = [] # TODO: Real one
                cpuRange["Input names"] = [] # TODO: Real one
            elif kind == "Optim":
                cpuRange["name"] = gpuRange["name"] = "torch.optim." + name # TODO: Remove "SGD.zero_grad"            
        else:
            '''
            parts = node.name.split("###")
            if len(parts) != 2:
                print("parts!=2:  " + node.name)
                return
            fullName = parts[0]
            eventId = parts[1]
            '''            

            cat = "Operator"            
            cpuRange["cat"] = gpuRange["cat"] = cat
            if fullName.find(",") >= 0:
                name = fullName[: fullName.find(",")]
            else:
                name = fullName
            cpuRange["name"] = gpuRange["name"] = name
            sizeStr = "sizes = "
            sizeStart = fullName.find(sizeStr)
            if sizeStart == -1:
                #print("No sizes: " + fullName)
                cpuRange["Input dims"] = "null"
            else:
                sizeStart += len(sizeStr)
                sizeValueStr = fullName[sizeStart:]
                cpuRange["Input dims"] = sizeValueStr # TODO: Not string type, libmongpu shows array type!
            # TODO: "seq"            

    if (cat == "Python" or cat == "Operator") and (not node.startTimeDevice is None):     
        WriteToChromeGpuRange(gpuRange, dstFile)        
        WriteGpuFlowToChrome(gpuRange, dstFile)
        WriteToChromeCpuRange(cpuRange, dstFile)    
        WriteCpuFlowToChrome(cpuRange, dstFile)
    elif (cat == "Runtime") and (not gpuRange is None):
        WriteToChromeGpuRange(gpuRange, dstFile)        
        WriteGpuFlowToChrome(gpuRange, dstFile)
        WriteToChromeCpuRange(cpuRange, dstFile)    
        WriteCpuFlowToChrome(cpuRange, dstFile)
    else:
        WriteToChromeCpuRange(cpuRange, dstFile)    


def WriteToChromeCpuRange(cpuRange, file):
    fields = ["ph", "cat", "ts", "dur", "pid", "tid", "name"]
    args = ["Input dims", "Input type", "Input names", "Device", "correlation", "stack", "Step"]
    WriteSpecificFieldsToChrome(cpuRange, fields, args, file)


def WriteToChromeGpuRange(gpuRange, file):
    fields = ["ph", "cat", "ts", "dur", "pid", "tid", "name"]
    args = ["Device", "context", "stream", "correlation", "bytes", "memory bandwidth (GB/s)"]
    WriteSpecificFieldsToChrome(gpuRange, fields, args, file)
    return


def WriteFields(rangeDict, fields, file):
    for field in fields:
        if rangeDict.__contains__(field):
            value = rangeDict[field]
            isString = isinstance(value, str)
            file.write('"' + field + '": ')
            if isString:
                file.write('"')
            file.write(str(value))
            if isString:
                file.write('"')
            file.write(',')

def WriteSpecificFieldsToChrome(range, fields, args, file): 
    file.write('{\n')    
    WriteFields(range, fields, file)
    file.write('"args": {\n')
    WriteFields(range["args"], args, file)
    file.seek(file.tell() - 1, os.SEEK_SET) # Remove the last ','
    file.write('}\n')
    file.write('},\n')        

def WriteFlow(flow, fields, file):
    file.write('{ ')
    WriteFields(flow, fields, file)
    file.seek(file.tell() - 1, os.SEEK_SET)  # Remove the last ','
    file.write('},\n')
    return

def WriteCpuFlowToChrome(cpuRange, file):
    cpuFlow = {}
    cpuFlow["ph"] = "s"
    cpuFlow["id"] = cpuRange["args"]["correlation"]
    cpuFlow["pid"] = cpuRange["pid"]
    cpuFlow["tid"] = cpuRange["tid"]
    cpuFlow["ts"] = cpuRange["ts"]
    cpuFlow["cat"] = "async"
    cpuFlow["name"] = "launch"
    cpuFields = ["ph", "id", "pid", "tid", "ts", "cat", "name"]
    WriteFlow(cpuFlow, cpuFields, file)

def WriteGpuFlowToChrome(gpuRange, file):
    gpuFlow = {}
    gpuFlow["ph"] = "f"
    gpuFlow["id"] = gpuRange["args"]["correlation"]
    gpuFlow["pid"] = gpuRange["pid"]
    gpuFlow["tid"] = gpuRange["tid"]
    gpuFlow["ts"] = gpuRange["ts"]
    gpuFlow["cat"] = "async"
    gpuFlow["name"] = "launch"
    gpuFlow["bp"] = "e"
    gpuFields = ["ph", "id", "pid", "tid", "ts", "cat", "name", "bp"]
    WriteFlow(gpuFlow, gpuFields, file)


def PrintCounts(roots):
    for tid, root in roots.items():
        name2count = {}
        group_by_name(root, name2count)
        for key, value in name2count.items():
            print(key + "," + str(value))
        print("\n\n\n==================\n\n\n")

def group_by_name(node, name2count):
    if isinstance(node, TreeNode):
        if node.name.startswith("aten::"):
            if not name2count.__contains__("aten::"):
                name2count["aten::"] = 0
            name2count["aten::"] += 1
        else:
            name = node.name.split(',')[0]
            name = name.split('###')[0]
            if not name2count.__contains__(name):
                name2count[name] = 0
            name2count[name] += 1
        for child in node.children:
            group_by_name(child, name2count)
    else:
        print("node is not TreeNode:" + str(node))

def RemoveRuntimeNotInStep(roots):
    for tid, root in roots.items():
        new_children_list = []
        for child in root.children:
            if not isinstance(child, TreeNodeRuntime):
                new_children_list.append(child)
        root.children = new_children_list

# ===================== main ========================
def main():    
    args = parseArgs()    
    db = DB(args.file)

    # Build the call tree.
    cudaDict = buildCUDADict(db)
    
    roots = {} # <Thread Id, Tree Root>
    buildNVTXTree(db, roots)
    buildRuntimeTree(db, roots, cudaDict)    

    RemoveRuntimeNotInStep(roots)
    
    for tid, root in roots.items():
        root.name = 'Root'
        root.Summarize()
    print("Summarize() finished.")
    
    export_chrome_trace("./chrome_dump.txt", roots)
    db.close()
    return

class CommonVars:
    def __init__(self):
        self.correlationId = 0

if __name__ == '__main__':
    main()
