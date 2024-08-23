project(
    'isoflex', 
    'c',
    version: '0.0.1',
    meson_version: '>= 1.0',)

name = 'isoflex'

platform = host_machine.system()

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()


inc_np = include_directories(incdir_numpy)

install_subdir(name, install_dir: py.get_install_dir() / name, strip_directory: true)