"""3D mesh manipulation utilities."""

from builtins import str
from collections import OrderedDict
import numpy as np
from transforms3d import quaternions
from transforms3d.quaternions import axangle2quat, mat2quat
import os.path as osp

cur_dir = osp.dirname(osp.abspath(__file__))


def frustum(left, right, bottom, top, znear, zfar):
    """Create view frustum matrix."""
    assert right != left
    assert bottom != top
    assert znear != zfar

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[2, 0] = (right + left) / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[3, 1] = (top + bottom) / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
    M[2, 3] = -1.0
    return M


def perspective(fovy, aspect, znear, zfar):
    """Create perspective projection matrix."""
    assert znear != zfar
    h = np.tan(fovy / 360.0 * np.pi) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)


def anorm(x, axis=None, keepdims=False):
    """Compute L2 norms alogn specified axes."""
    return np.sqrt((x * x).sum(axis=axis, keepdims=keepdims))


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(anorm(v, axis=axis, keepdims=True), eps)


def lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt modelview matrix."""
    eye = np.float32(eye)
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)
    M = np.eye(4, dtype=np.float32)
    R = M[:3, :3]
    R[:] = [side, up, -forward]
    M[:3, 3] = -(R.dot(eye))
    return M


def sample_view(min_dist, max_dist=None):
    """Sample random camera position.

    Sample origin directed camera position in given distance range from
    the origin. ModelView matrix is returned.
    """
    if max_dist is None:
        max_dist = min_dist
    dist = np.random.uniform(min_dist, max_dist)
    eye = np.random.normal(size=3)
    eye = normalize(eye) * dist
    return lookat(eye)


def homotrans(M, p):
    p = np.asarray(p)
    if p.shape[-1] == M.shape[1] - 1:
        p = np.append(p, np.ones_like(p[..., :1]), -1)
    p = np.dot(p, M.T)
    return p[..., :-1] / p[..., -1:]


def _parse_vertex_tuple(s):
    """Parse vertex indices in '/' separated form (like 'i/j/k', 'i//k'.

    ...).
    """
    vt = [0, 0, 0]
    for i, c in enumerate(s.split("/")):
        if c:
            vt[i] = int(c)
    return tuple(vt)


def _unify_rows(a):
    """Unify lengths of each row of a."""
    lens = np.fromiter(map(len, a), np.int32)
    if not (lens[0] == lens).all():
        out = np.zeros((len(a), lens.max()), np.float32)
        for i, row in enumerate(a):
            out[i, : lens[i]] = row
    else:
        out = np.float32(a)
    return out


def loadTexture(path):
    from PIL import Image
    import OpenGL.GL as GL

    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.fromstring(img.tobytes(), np.uint8)
    # print(img_data.shape)
    width, height = img.size
    # glTexImage2D expects the first element of the image data to be the
    # bottom-left corner of the image.  Subsequent elements go left to right,
    # with subsequent lines going from bottom to top.

    # However, the image data was created with PIL Image tostring and numpy's
    # fromstring, which means we have to do a bit of reorganization. The first
    # element in the data output by tostring() will be the top-left corner of
    # the image, with following values going left-to-right and lines going
    # top-to-bottom.  So, we need to flip the vertical coordinate (y).
    texture = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB,
        width,
        height,
        0,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        img_data,
    )
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return texture


def shader_from_path(shader_filename):
    shader_path = osp.join(cur_dir, "../shader", shader_filename)
    assert osp.exists(shader_path)
    with open(shader_path, "r") as f:
        return f.read()


def load_obj(fn):
    """Load 3d mesh form .obj' file.

    Args:
      fn: Input file name or file-like object.

    Returns:
      dictionary with the following keys (some of which may be missing):
        position: np.float32, (n, 3) array, vertex positions
        uv: np.float32, (n, 2) array, vertex uv coordinates
        normal: np.float32, (n, 3) array, vertex uv normals
        face: np.int32, (k*3,) traingular face indices
    """
    position = [np.zeros(3, dtype=np.float32)]
    normal = [np.zeros(3, dtype=np.float32)]
    uv = [np.zeros(2, dtype=np.float32)]

    tuple2idx = OrderedDict()
    trinagle_indices = []

    input_file = open(fn) if isinstance(fn, str) else fn
    for line in input_file:
        line = line.strip()
        if not line or line[0] == "#":
            continue
        line = line.split(" ", 1)
        tag = line[0]
        if len(line) > 1:
            line = line[1]
        else:
            line = ""
        if tag == "v":
            position.append(np.fromstring(line, sep=" "))
        elif tag == "vt":
            uv.append(np.fromstring(line, sep=" "))
        elif tag == "vn":
            normal.append(np.fromstring(line, sep=" "))
        elif tag == "f":
            output_face_indices = []
            for chunk in line.split():
                # tuple order: pos_idx, uv_idx, normal_idx
                vt = _parse_vertex_tuple(chunk)
                if vt not in tuple2idx:  # create a new output vertex?
                    tuple2idx[vt] = len(tuple2idx)
                output_face_indices.append(tuple2idx[vt])
            # generate face triangles
            for i in range(1, len(output_face_indices) - 1):
                for vi in [0, i, i + 1]:
                    trinagle_indices.append(output_face_indices[vi])

    outputs = {}
    outputs["face"] = np.int32(trinagle_indices)
    pos_idx, uv_idx, normal_idx = np.int32(list(tuple2idx)).T
    if np.any(pos_idx):
        outputs["position"] = _unify_rows(position)[pos_idx]
    if np.any(uv_idx):
        outputs["uv"] = _unify_rows(uv)[uv_idx]
    if np.any(normal_idx):
        outputs["normal"] = _unify_rows(normal)[normal_idx]
    return outputs


def normalize_mesh(mesh):
    """Scale mesh to fit into -1..1 cube."""
    mesh = dict(mesh)
    pos = mesh["position"][:, :3].copy()
    pos -= (pos.max(0) + pos.min(0)) / 2.0
    pos /= np.abs(pos).max()
    mesh["position"] = pos
    return mesh


def quat2rotmat(quat):
    quat_mat = np.eye(4)
    quat_mat[:3, :3] = quaternions.quat2mat(quat)
    return quat_mat


def mat2rotmat(mat):
    quat_mat = np.eye(4)
    quat_mat[:3, :3] = mat
    return quat_mat


def xyz2mat(xyz):
    trans_mat = np.eye(4)
    trans_mat[-1, :3] = xyz
    return trans_mat


def mat2xyz(mat):
    xyz = mat[-1, :3]
    xyz[np.isnan(xyz)] = 0
    return xyz


def safemat2quat(mat):
    quat = np.array([1, 0, 0, 0])
    try:
        quat = mat2quat(mat)
    except:
        pass
    quat[np.isnan(quat)] = 0
    return quat
