from importlib.resources import files

project_name = 'tabled'
root_posix_path = files(project_name)


def local_posix(*path):
    return root_posix_path.joinpath(*path)


def local_text(*path):
    return local_posix(*path).read_text()


test_data_path = ('tests', 'data')
test_data_posix = local_posix(*test_data_path)


def test_data(name, text=False):
    name_posix = test_data_posix.joinpath(name)
    if not text:
        return name_posix.read_bytes()
    else:
        return name_posix.read_text()
