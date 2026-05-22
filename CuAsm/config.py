# -*- coding: utf-8 -*-

from elftools.elf.structs import ELFStructs
import os
import shutil

# TODO: log system

def getDefaultStruct(st):
    return st.parse(b'\x00'*st.sizeof())


def _resolve_nvdisasm():
    """Locate nvdisasm portably instead of hardcoding one install's path.

    Order: explicit NVDISASM_PATH env -> nvdisasm on PATH -> CUDA_HOME/CUDA_PATH
    bin -> common /usr/local/cuda location. Falls back to the bare name so the
    error (if any) surfaces at call time rather than import time.
    """
    env = os.environ.get('NVDISASM_PATH')
    if env and os.path.isfile(env):
        return env
    found = shutil.which('nvdisasm')
    if found:
        return found
    for root in (os.environ.get('CUDA_HOME'), os.environ.get('CUDA_PATH'),
                 '/usr/local/cuda'):
        if root:
            cand = os.path.join(root, 'bin', 'nvdisasm')
            if os.path.isfile(cand):
                return cand
    return 'nvdisasm'


class Config(object):

    # Path to nvdisasm — resolved from env/PATH/CUDA_HOME (see _resolve_nvdisasm).
    NVDISASM_PATH = _resolve_nvdisasm()

    # Currently only little_endian and ELF64 is supported
    # NOTE: There are quite a lot of hardcodes for endianness and elfclass
    #       thus just modifying the value here will not work
    CubinELFStructs = ELFStructs(little_endian=True, elfclass=64)
    CubinELFStructs.create_basic_structs()
    CubinELFStructs.create_advanced_structs()

    defaultCubinFileHeader = CubinELFStructs.Elf_Ehdr.parse(bytes.fromhex(''.join([
                            '7f454c460201013307000000000000000200be00650000000000000000000000',
                            'c09000000000000000890000000000004b054b0040003800030040001f000100'])))

    # 'e_phentsize': 56, 'e_shentsize': 64
    defaultSectionHeader = getDefaultStruct(CubinELFStructs.Elf_Shdr)
    defaultSegmentHeader = getDefaultStruct(CubinELFStructs.Elf_Phdr)
    
    # 24 B
    defaultSymbol = getDefaultStruct(CubinELFStructs.Elf_Sym)

    # rel/rela
    defaultRel    = getDefaultStruct(CubinELFStructs.Elf_Rel)
    defaultRela   = getDefaultStruct(CubinELFStructs.Elf_Rela)

    # TODO: load from / save to file?
    def load(self):
        pass

    def save(self):
        pass

    @staticmethod
    def getDefaultInsAsmReposFile(version_number):
        module_dir = os.path.split(__file__)
        repos_dir = os.path.join(module_dir[0], 'InsAsmRepos')
        repos_name = 'DefaultInsAsmRepos.sm_%d.txt' % version_number
        repos_path = os.path.join(repos_dir, repos_name)
        return repos_path

    @staticmethod
    def getDefaultIOInfoFile(version_number):
        module_dir = os.path.split(__file__)
        fdir = os.path.join(module_dir[0], 'InsAsmRepos')
        
        fname = 'IOInfo.sm_%d.txt' % version_number
        fpath = os.path.join(fdir, fname)
        
        if not os.path.isfile(fpath):
            fpath = os.path.join(fdir, 'IOInfo.all.json')
        
        return fpath
        
