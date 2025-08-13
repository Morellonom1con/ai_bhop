from collections import UserList


class Path(UserList):
    """Full path manipulation in a nested object.
    
    A path is a full (starts with '/') address in a nested object.
    Items at each level are separated by '/'.
    """
    def __init__(self, items=None):
        if items is None:
            items = []
        else:
            if isinstance(items, Path):
                items = items.data
            elif isinstance(items, str):
                if items[0] != '/':
                    raise ValueError('Path class only supports absolute paths')
                items = items.strip('/').split('/')
                if items == ['']: items = []
        super().__init__(items)

    def __str__(self):
        return '/' + '/'.join(self)

    def __repr__(self):
        return f'<Path({str(self)})>'

    def __truediv__(self, other):
        return Path(self + Path('/' + str(other)))

    @property
    def parent(self):
        """Get parent of path. Equivalent to `dirname`.
        
        If path is /, raise IndexError by analogy with [].pop()
        """
        if len(self) == 0:
            raise IndexError("Root '/' has no parent")
        return Path('/' + '/'.join(self[:-1]))

    def startswith(self, other):
        """Check if path contains other from root /"""
        return str(self).startswith(str(other))

    def split(self):
        """Like os.path.split, return a tuple of (parent path, last key)"""
        if len(self) == 0:
            raise TypeError(".split() cannot be called on root Path('/')")
        return (self.parent, self[-1])