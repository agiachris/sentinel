import torch
from torch import Tensor
from typing import Optional


def KORNIA_CHECK_SHAPE(x: Tensor, shape: list[str], raises: bool = True) -> bool:
    """Check whether a tensor has a specified shape.

    The shape can be specified with a implicit or explicit list of strings.
    The guard also check whether the variable is a type `Tensor`.

    Args:
        x: the tensor to evaluate.
        shape: a list with strings with the expected shape.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        Exception: if the input tensor is has not the expected shape and raises is True.

    Example:
        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])  # implicit
        True

        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["2", "3", "H", "W"])  # explicit
        True
    """
    if "*" == shape[0]:
        shape_to_check = shape[1:]
        x_shape_to_check = x.shape[-len(shape) + 1 :]
    elif "*" == shape[-1]:
        shape_to_check = shape[:-1]
        x_shape_to_check = x.shape[: len(shape) - 1]
    else:
        shape_to_check = shape
        x_shape_to_check = x.shape

    if len(x_shape_to_check) != len(shape_to_check):
        if raises:
            raise TypeError(f"{x} shape must be [{shape}]. Got {x.shape}")
        else:
            return False

    for i in range(len(x_shape_to_check)):
        # The voodoo below is because torchscript does not like
        # that dim can be both int and str
        dim_: str = shape_to_check[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            if raises:
                raise TypeError(f"{x} shape must be [{shape}]. Got {x.shape}")
            else:
                return False
    return True


def create_meshgrid(
    height: int,
    width: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # generate grid by stacking coordinates
    base_grid: Tensor = torch.stack(
        torch.meshgrid([xs, ys], indexing="ij"), dim=-1
    )  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def normalize_points_with_intrinsics(point_2d: Tensor, camera_matrix: Tensor) -> Tensor:
    """Normalizes points with intrinsics. Useful for conversion of keypoints to be used with essential matrix.

    Args:
        point_2d: tensor containing the 2d points in the image pixel coordinates. The shape of the tensor can be
                  :math:`(*, 2)`.
        camera_matrix: tensor containing the intrinsics camera matrix. The tensor shape must be :math:`(*, 3, 3)`.

    Returns:
        tensor of (u, v) cam coordinates with shape :math:`(*, 2)`.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> X = torch.rand(1, 2)
        >>> K = torch.eye(3)[None]
        >>> normalize_points_with_intrinsics(X, K)
        tensor([[0.4963, 0.7682]])
    """
    # KORNIA_CHECK_SHAPE(point_2d, ["*", "2"])
    # KORNIA_CHECK_SHAPE(camera_matrix, ["*", "3", "3"])

    # projection eq. K_inv * [u v 1]'
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy

    # unpack coordinates
    u_coord: Tensor = point_2d[..., 0]
    v_coord: Tensor = point_2d[..., 1]

    # unpack intrinsics
    fx: Tensor = camera_matrix[..., 0, 0]
    fy: Tensor = camera_matrix[..., 1, 1]
    cx: Tensor = camera_matrix[..., 0, 2]
    cy: Tensor = camera_matrix[..., 1, 2]

    # projective
    x_coord: Tensor = (u_coord - cx) / fx
    y_coord: Tensor = (v_coord - cy) / fy

    xy: Tensor = torch.stack([x_coord, y_coord], dim=-1)
    return xy


def unproject_points(
    point_2d: torch.Tensor, depth: torch.Tensor, camera_matrix: torch.Tensor
) -> torch.Tensor:
    r"""Unproject a 2d point in 3d.

    Transform coordinates in the pixel frame to the camera frame.

    Args:
        point2d: tensor containing the 2d to be projected to
            world coordinates. The shape of the tensor can be :math:`(*, 2)`.
        depth: tensor containing the depth value of each 2d
            points. The tensor shape must be equal to point2d :math:`(*, 1)`.
        camera_matrix: tensor containing the intrinsics camera
            matrix. The tensor shape must be :math:`(*, 3, 3)`.
        normalize: whether to normalize the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position.

    Returns:
        tensor of (x, y, z) world coordinates with shape :math:`(*, 3)`.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> x = torch.rand(1, 2)
        >>> depth = torch.ones(1, 1)
        >>> K = torch.eye(3)[None]
        >>> unproject_points(x, depth, K)
        tensor([[0.4963, 0.7682, 1.0000]])
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depth type is not a torch.Tensor. Got {type(depth)}")

    if not depth.shape[-1] == 1:
        raise ValueError(
            f"Input depth must be in the shape of (*, 1). Got {depth.shape}"
        )

    xy: torch.Tensor = normalize_points_with_intrinsics(point_2d, camera_matrix)
    xyz: torch.Tensor = torch.nn.functional.pad(xy, [0, 1], "constant", 1.0)

    return xyz * depth


def depth_to_3d(depth: Tensor, camera_matrix: Tensor, mask: Tensor) -> Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    .. note::

        This is an alternative implementation of `depth_to_3d` that does not require the creation of a meshgrid.
        In future, we will support only this implementation.

    Args:
        depth: image tensor containing a depth value per pixel with shape :math:`(B, 1, H, W)`.
        camera_matrix: tensor containing the camera intrinsics with shape :math:`(B, 3, 3)`.

    Return:
        tensor with a 3d point per pixel of the same resolution as the input :math:`(B, 3, H, W)`.

    Example:
        >>> depth = torch.rand(1, 1, 4, 4)
        >>> K = torch.eye(3)[None]
        >>> depth_to_3d(depth, K).shape
        torch.Size([1, 3, 4, 4])
    """
    if not isinstance(depth, Tensor):
        raise TypeError(f"Input depht type is not a Tensor. Got {type(depth)}.")

    if not (len(depth.shape) == 4 and depth.shape[-3] == 1):
        raise ValueError(
            f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}"
        )

    if not isinstance(camera_matrix, Tensor):
        raise TypeError(
            f"Input camera_matrix type is not a Tensor. Got {type(camera_matrix)}."
        )

    if not (len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3)):
        raise ValueError(
            f"Input camera_matrix must have a shape (B, 3, 3). Got: {camera_matrix.shape}."
        )

    if mask is None:
        # create base coordinates grid
        _, _, height, width = depth.shape
        points_2d: Tensor = create_meshgrid(height, width)  # 1xHxWx2
        points_2d = points_2d.to(depth.device).to(depth.dtype)

        # depth should come in Bx1xHxW
        depth = torch.where(depth > 1e-3, (0.12 * camera_matrix[0, 0, 0]) / depth, 0.0)
        points_depth: Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

        # project pixels to camera frame
        camera_matrix_tmp: Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
        points_3d: Tensor = unproject_points(
            points_2d, points_depth, camera_matrix_tmp
        )  # BxHxWx3

        return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW
    else:
        # tt = time.time()
        batch_size, _, height, width = depth.shape
        assert batch_size == 1, "masking mode only support B = 1"

        # filter depth
        filtered_depths = depth.view(-1, 1)[mask]
        filtered_depths = torch.where(
            filtered_depths > 1e-3,
            (0.12 * camera_matrix[0, 0, 0]) / filtered_depths,
            0.0,
        )
        # elapsed = time.time() - tt
        # print(f"  [depth_to_3d] filter depth takes {elapsed:.3f}s")
        # tt = time.time()

        # create base coordinates grid
        points_x = torch.remainder(mask, width)  # idxs % w
        points_y = torch.floor_divide(mask, width)  # idxs // w
        points_2d = torch.column_stack((points_x, points_y))  # (N, 2)
        # elapsed = time.time() - tt
        # print(f"  [depth_to_3d] create base coordinates grid takes {elapsed:.3f}s")
        # tt = time.time()

        # project pixels to camera frame
        points_3d: Tensor = unproject_points(points_2d, filtered_depths, camera_matrix)
        # elapsed = time.time() - tt
        # print(f"  [depth_to_3d] unproject points takes {elapsed:.3f}s")
        return points_3d  # (N, 3)


def make_pointcloud(cam_matrix, disparity, mask=None):
    # tt = time.time()
    if not torch.is_tensor(cam_matrix):
        cam_matrix_torch = torch.from_numpy(cam_matrix[None]).to(disparity.device)
    else:
        cam_matrix_torch = cam_matrix[None]
    pc = depth_to_3d(disparity, cam_matrix_torch, mask)
    # elapsed = time.time() - tt
    # print(f"[depth_to_3d] make_pointcloud takes {elapsed:.3f}s")
    return pc
