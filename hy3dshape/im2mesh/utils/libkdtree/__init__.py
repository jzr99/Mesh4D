try:
    from .pykdtree.kdtree import KDTree
except Exception:
    print("Use external pykdtree package")
    from pykdtree.kdtree import KDTree 


__all__ = [
    KDTree
]
