import logging

import numpy as np

logger = logging.getLogger(__name__)


def npa_to_psg(npa, path, object_type, object_id=None):
    """
    PSG format version 2
    To convert a numpy array (2d or 3d) to a segmentation file
    Later version might support 4d as well
    Note that for interchange between this code and backend, 3D x and y need to be swapped

    Negative values won't have any effect, logger calls info statement for 'empty'
    arrays and 'empty' labels

    :param npa: numpy array to convert (2d or 3d): any higher then zero values will be 'high',
    zero or lower be 'low'
    :param path: complete path inc. file (str)
    :param object_type: structure label L1, max. 255 chr. (str)
    :param object_id: structure label L2, max. 255 chr. (str)
    :return: Nothing
    """
    __VERSION__ = 2

    logger.debug("Converting numpy array to PSG format")

    npa = np.array(npa)
    npa = npa > 0

    if npa.sum() == 0:
        logger.warning("Writing PSG with no voxels set to high")

    labels_fit = True
    object_id = "" if object_id is None else object_id
    if len(object_type) > 255 or len(object_id) > 255:
        logger.error("npa_to_psg: either of both labels is larger then 255 chr.")
        labels_fit = False
    assert labels_fit

    dims_are_supported = True
    if len(npa.shape) == 3:
        npa = np.swapaxes(npa, 0, 1)
        (dim_x, dim_y, dim_z) = npa.shape
    elif len(npa.shape) == 2:
        npa = np.swapaxes(npa, 0, 1)
        (dim_x, dim_y) = npa.shape
        dim_z = 0
    else:
        logger.error(
            "npa_to_psg: shape of npa is not two- or three-dimensional, "
            "shape: {}".format(npa.shape)
        )
        dims_are_supported = False
    assert dims_are_supported

    if object_type == "":
        logger.debug("Writing PSG with blank object_type")
    if object_id == "":
        logger.debug("Writing PSG with blank object_id")

    logger.debug("Generating PSG file {}".format(path))

    with open(path, "wb") as f:
        b_version = __VERSION__.to_bytes(1, byteorder="big", signed=False)
        b_dim_x = dim_x.to_bytes(2, byteorder="big", signed=False)
        b_dim_y = dim_y.to_bytes(2, byteorder="big", signed=False)
        b_dim_z = dim_z.to_bytes(2, byteorder="big", signed=False)
        header_mutable_bytes = [b_version, b_dim_x, b_dim_y, b_dim_z]

        for byte in header_mutable_bytes:
            f.write(byte)

        b_object_type_length = len(object_type).to_bytes(
            1, byteorder="big", signed=False
        )
        f.write(b_object_type_length)
        b_object_type = object_type.encode("utf-8")
        f.write(b_object_type)
        b_object_id_length = len(object_id).to_bytes(1, byteorder="big", signed=False)
        f.write(b_object_id_length)
        b_object_id = object_id.encode("utf-8")
        f.write(b_object_id)

        npa = npa.flatten()

        npa_shift = npa[:-1]
        npa_shift = np.insert(npa_shift, 0, 0)
        npa_diff_where = np.where(npa != npa_shift)[0]
        for num in npa_diff_where:
            data_point = int(num).to_bytes(4, byteorder="big", signed=False)
            f.write(data_point)


def psg_to_npa(path):
    """
    PSG format version 2
    To convert a segmentation file to numpy array (2d or 3d)
    Later version might support 4d as well
    Note that for interchange between this code and backend, 3D x and y need to be swapped

    :param path: complete path inc. file (str)
    :return: npa (np array), object_type label (str), object_id label (str)
    """
    logger.debug(f"Converting {path} from PSG format to numpy array and labels")

    header_chunksizes = [1, 2, 2, 2]
    header_values = []

    with open(path, "rb") as f:
        for header_chunksize in header_chunksizes:
            chunk = f.read(header_chunksize)
            i = int.from_bytes(chunk, byteorder="big")
            header_values.append(i)
        for i in range(2):
            string_size = int.from_bytes(f.read(1), byteorder="big")
            str_encoded = f.read(string_size)
            string = str_encoded.decode("utf-8")
            header_values.append(string)

        version_is_supported = True
        if not header_values[0] == 2:
            logger.error(
                "psg_to_npa: incorrect load function for file version, unsupported: {}"
                .format(header_values[0])
            )
            version_is_supported = False
        assert version_is_supported

        if header_values[3] == 0:
            npa = np.zeros((header_values[1] * header_values[2]), dtype=np.bool_)
        else:
            npa = np.zeros(
                (header_values[1] * header_values[2] * header_values[3]), dtype=np.bool_
            )

        while True:
            chunk_start = f.read(4)
            chunk_end = f.read(4)
            if chunk_start:
                index_start = int.from_bytes(chunk_start, byteorder="big")
            else:
                break
            if chunk_end:
                index_end = int.from_bytes(chunk_end, byteorder="big")
                npa[index_start:index_end] = True
            else:
                npa[index_start:] = True
                break

    if header_values[3] == 0:
        npa = np.reshape(npa, (header_values[1], header_values[2]))
        npa = np.swapaxes(npa, 0, 1)
    else:
        npa = np.reshape(npa, (header_values[1], header_values[2], header_values[3]))
        npa = np.swapaxes(npa, 0, 1)

    object_type = header_values[4]
    object_id = header_values[5]

    return npa, object_type, object_id
