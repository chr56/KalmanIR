def scan_and_import_modules(folder_path: str, package: str, filename_suffix: str):
    """ Automatically scan and import modules:
        scan all the files under the 'folder_path'
        and collect files ending with '_filename_suffix.py' then import them
    :param folder_path: folder path to be scanned
    :param package: package of folder to be scanned
    :param filename_suffix: filename suffix that should be imported
    """
    from os import path as osp
    from basicsr.utils import scandir
    filenames = [
        osp.splitext(osp.basename(v))[0] for v in scandir(folder_path)
        if v.endswith(f'_{filename_suffix}.py')
    ]
    import importlib
    return [
        importlib.import_module(f'{package}.{file_name}')
        for file_name in filenames
    ]


def dynamic_instantiation(modules: list, cls_type: str, opt: dict):
    """Dynamically instantiate class.

    :param modules: List of modules from importlib files
    :param cls_type: Type of class to be instantiated
    :param opt: Class initialization kwargs.

    :returns: Instantiated class
    """

    cls_ = None
    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')

    return cls_(**opt)
