'''
   AUTO

   Use JSON as a "Execution specification"
'''

from ase.build import bulk
from ase.io import read
from gpaw import GPAW
from gpaw.utilities import h2gpts
from gpaw.mixer import BroydenMixer
from gpaw.mixer import Mixer

#Spacegroup
import spglib

import json
from pprint import pprint

#id generation
import time

#set proc name
from setproctitle import *
import os

#a better range
import itertools

def frange(start, end=None, inc=1.0):
 "An xrange-like generator which yields float values"
 # imitate range/xrange strange assignment of argument meanings
 if end is None:
  end = start + 0.0     # Ensure a float value for 'end'
  start = 0.0

 assert inc                # sanity check

 for i in itertools.count( ):
  next = start + i * inc

  if (inc>0.0 and next>=end) or (inc<0.0 and next<=end):
    break
  yield next

### CLASSES ###
class execution(object):
 
 def __init__(self, executionInput, xyz=None, stateFile='state.json'):
  '''
  Setup a new execution

  Name:  input file
  xyz:   xyz coordinates (for future automated execution)
  stateFile: state file to use for tracking of executions

  TODO: 

  - only ground state in non-periodic cell supported NOW, more to add!
  - inputs from a dict to better handle automated executions
  '''
  if isinstance(executionInput, str):
   self.inp = self.json(executionInput)
   self.inputFileName = executionInput

  if isinstance(executionInput, dict):
   self.inp = executionInput
   self.inputFileName = 'dict' # TODO: generate meaningful name here (how?)

  self.setup()

 def setup(self):
  '''setup'''

  self.functional = self.inp.get('functional') or 'LDA'
  self.runtype = self.inp.get('runtype') or 'calc'

  # this actually can't be none!
  _xyz = self.inp.get('xyz')

  if isinstance(_xyz, str):
   self.inputXYZ = _xyz
   self.atoms = read(self.inputXYZ)
  else:
   # support reading frm a dict (TODO: other type of objects - ?)
   from ase.atoms import Atoms
   from ase.atom  import Atom

   self.atoms = \
    Atoms([Atom(a[0],(a[1],a[2],a[3])) for a in _xyz])

  # TODO: support multiple cell types
  self.atoms.cell=(self.inp['cell']['x'],
                   self.inp['cell']['y'],
                   self.inp['cell']['z'])

  # Mixer setup - only "broyden" supported
  self.mixer = self.inp.get('mixer')

  if self.mixer is None:
   self.mixer = Mixer
  else:
   self.mixer = BroydenMixer

  # Custom ID: allow to trace parameter set using a custom id
  self.custom_id = self.inp.get('custom_id')

  # Periodic Boundary Conditions
  self.atoms.pbc=False

  # This should not be done always
  self.atoms.center()

  self.nbands = self.inp.get('nbands')

  self.gpts = None
  self.setup_gpts()

  self.max_iterations = self.inp.get('maxiter') or 333;

  # Name: to be set by __str__ first time it is run
  self.name = None

  # 
  self.logExtension='txt'
  self.dataExtension='gpw'

  # Timing / statistics
  self.lastElapsed = None

 #
 def get_optimizer(self,name='QuasiNewton'):

  from importlib import import_module

  name = name.lower() 
  _optimMap = \
   {
    'mdmin':       'MDMin',
    'hesslbfgs':   'HessLBFGS',
    'linelbfgs':   'LineLBFGS',
    'fire':        'FIRE',
    'lbfgs':       'LBFGS',
    'lbfgsls':     'LBFGSLineSearch',
    'bfgsls':      'BFGSLineSearch',
    'bfgs':        'BFGS',
    'quasinewton': 'QuasiNewton'
   }
  _class = _optimMap[name]
  _m = import_module('ase.optimize')
  return getattr(_m,_class)

 #
 def setup_gpts(self):
  self.gpts=h2gpts(self.inp.get('spacing') or 0.20,
                   self.atoms.get_cell(),
                   idiv=self.inp.get('idiv') or 8)

 def filename(self, force_id=None, extension='txt'):
  '''to correctly support this:
     stateFile support must be completed
     as it isn't just add "_0" to __str__() for now
  '''
  base = self.__str__()

  if self.custom_id is not None:
   base = base + '_' + self.custom_id

  if force_id is None:
   _fn = base + '_0.' + extension
  else:
   _fn = '{}_{}.{}'.format(base,force_id,extension)
  
  return _fn

 #
 @property
 def xyz(self):
  return self.inp['xyz']

 #
 @property
 def spacegroup(self):
  '''international ID of determined spacegroup'''
  sg = spglib.get_symmetry_dataset(spglib.refine_cell(self.atoms))['international']

  return sg.replace('/',':')

 #
 def __str__(self):
  '''get execution name - to be used for log filename

  Components:

  - Number atoms
  - Volume
  - Spacegroup number
  - Functional name

  '''

  if self.name is None:
   self.name = '{}_{}_{}_{}'.format(len(self.atoms),
                             int(self.atoms.get_volume()*100),
                             self.spacegroup,
                             self.functional)

  _gpaw = os.environ['GPAW']
  _pwd = os.environ['PWD']

  self.procname='{} ({})'.format(self.name, _pwd.replace(_gpaw+'/',''))
  setproctitle(self.procname)

  return self.name

 ##
 def print(self, *args, **kwargs):
  '''print wrapper - initial step before logger'''
  r = print(*args, *kwargs)
  return r

 #
 def pretty(self):
  '''pretty print original parameters'''
  _p=pformat(self.inp)
  self.print(_p)

 #
 def json(self,filename):
  '''load parameters from json'''

  with open(filename,'rb') as f:
   s = f.read()
   j = json.loads(s)

  return j

 #
 def run(self, force_id=None, runtype=None):
  '''
    perform execution - only calc or optim supported now
  '''

  _lastStart = time.time()

  if runtype is None and self.runtype is 'calc':
   _f = self.calc
  else:
   _f = self.optim

  # calculate
  try:
   result = _f(force_id)
  finally:
   _lastEnd = time.time()
   self.lastElapsed = _lastEnd - _lastStart

  return result

 #
 def calc(self, force_id=None, functional=None):
  '''prepare and launch execution
     force_id: use this id for filename generation
  '''

  # Allow to override the funtional
  _fnl = functional or self.functional

  # generate filenames
  _fn = self.filename(force_id)
  _dn = self.filename(force_id, extension=self.dataExtension)

  self.print('Calculation with {} log and Input: {}.'.format(_fn, self.inputFileName))

  if self.nbands is None:
   _calc = GPAW(
                xc=_fnl,
                txt=self.filename(force_id),
                mixer=self.mixer(),
                maxiter=self.max_iterations,
                gpts=self.gpts
               )
  else:
   _calc = GPAW(
                xc=_fnl,
                txt=self.filename(force_id),
                nbands=self.nbands,
                mixer=self.mixer(),
                maxiter=self.max_iterations,
                gpts=self.gpts
               )


  self.atoms.set_calculator(_calc)
  self.atoms.get_potential_energy()

  # write out wavefunctions
  _calc.write(_dn, 'all')
  return _calc

 #####
 def optim(self, force_id=None,functional=None):
  '''prepare and launch execution
     force_id: use this id for filename generation
  '''

  _optim_name = self.inp.get('optimizer') or 'QuasiNewton'
  optimizer = self.get_optimizer(_optim_name)

  # Allow to override the funtional
  _fnl = functional or self.functional

  # generate filenames
  _fn = self.filename(force_id)
  _dn = self.filename(force_id, extension=self.dataExtension)

  self.print('Optimization with {} log and Input: {}.'.format(_fn, self.inputFileName))

  if self.nbands is None:
   _calc = GPAW(
                xc=_fnl,
                txt=self.filename(force_id),
                mixer=self.mixer(),
                maxiter=self.max_iterations,
                gpts=self.gpts
               )
  else:
   _calc = GPAW(
                xc=_fnl,
                txt=self.filename(force_id),
                nbands=self.nbands,
                mixer=self.mixer(),
                maxiter=self.max_iterations,
                gpts=self.gpts
               )

  self.atoms.set_calculator(_calc)

  self.opt = optimizer(self.atoms, trajectory=self.__str__()+'_emt.traj')
  self.opt.run(fmax=0.05)

  # write out wavefunctions
  _calc.write(_dn, 'all')
  return _calc

###
def main(argv):

 try:
  inputFile = argv[1]
 except:
  inputFile = 'input.json'

 inp = execution(inputFile)

 print(inp)
 try:
  inp.run(str(int(time.time())))
 except Exception as e:
  print('*EXCEPTION: '+str(e))
 finally:
  if inp.lastElapsed is not None:
   print('  ===> Elpased ' + str(inp.lastElapsed))

## call main

if __name__ == '__main__':
 import sys
 main(sys.argv)

## EOF ##
