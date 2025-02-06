import usb.core
import usb.util


def get_serial_number():
    VENDOR_ID = 0x046D
    PRODUCT_ID = 0x0843

    device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    assert device is not None, "Cannot find logitech device"

    serial_number = usb.util.get_string(device, device.iSerialNumber)
    return serial_number
    # return "SN0001"
