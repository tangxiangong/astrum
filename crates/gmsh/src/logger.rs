use crate::bindings::*;
use crate::{GError, GResult};
use std::ffi::*;

/// API: gmsh::logger::getLastError
/// Return last error message, if any.
/// ```c
/// GMSH_API void gmshLoggerGetLastError(char ** error,
///    int * ierr);
/// ```
/// ```c++
/// void getLastError(std::string & error);
/// ```
pub fn get_last_error() -> GResult<String> {
    let mut error_ptr = std::ptr::null_mut();
    let mut ierr = 0;
    unsafe {
        gmshLoggerGetLastError(&mut error_ptr, &mut ierr);

        let result = match (ierr, error_ptr.is_null()) {
            // 0 表示获取信息成功
            (0, false) => {
                // 安全：已确认指针非空
                let cstr = CStr::from_ptr(error_ptr);
                Ok(cstr.to_string_lossy().into_owned())
            }
            (0, true) => Err(GError::InvalidState(
                "Null pointer with success code".to_owned(),
            )),
            (_, _) => Err(GError::ApiError(
                "Failed to retrieve error message".to_owned(),
            )),
        };

        if !error_ptr.is_null() {
            libc::free(error_ptr as *mut libc::c_void);
        }

        result
    }
}
