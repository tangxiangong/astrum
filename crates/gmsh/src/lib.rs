include!("bindings.rs");

mod error;
pub use error::*;

pub mod logger;

use crate::{GError, GResult, bindings::*};
use std::ffi::*;

/// A model entity in the Gmsh API is represented by two integers: its
/// dimension (dim = 0, 1, 2 or 3) and its tag (its unique, strictly positive
/// identifier). When dealing with multiple model entities of possibly
/// different dimensions, the entities are packed as a vector of (dim, tag)
/// integer pairs.
/// ```c++
/// typedef std::vector<std::pair<int, int> > vectorpair;
/// ```
pub struct VectorPair(pub Vec<(usize, usize)>);

/// All the functions in the Gmsh C API that return arrays allocate the
/// necessary memory with gmshMalloc(). These arrays should be deallocated
/// with gmshFree().
/// ```c
/// GMSH_API void gmshFree(void *p);
/// GMSH_API void *gmshMalloc(size_t n);
/// ```
#[allow(dead_code)]
pub struct APIAllocatedArray {
    ptr: *mut c_void,
    len: usize,
}

impl Drop for APIAllocatedArray {
    fn drop(&mut self) {
        unsafe {
            gmshFree(self.ptr);
        }
    }
}

/// API: gmsh::initialize
///
/// Initialize the Gmsh API. This must be called before any call to the other
/// functions in the API. If `argc' and `argv' (or just `argv' in Python or Julia)
/// are provided, they will be handled in the same way as the command line
/// arguments in the Gmsh app. If `readConfigFiles' is set, read system Gmsh
/// configuration files (gmshrc and gmsh-options). If `run' is set, run in the
/// same way as the Gmsh app, either interactively or in batch mode depending on
/// the command line arguments. If `run' is not set, initializing the API sets the
/// options "General.AbortOnError" to 2 and "General.Terminal" to 1.
/// ```c++
/// GMSH_API void initialize(int argc = 0, char ** argv = 0,
/// const bool readConfigFiles = true,
/// const bool run = false);
/// ```
/// ```c
/// GMSH_API void gmshInitialize(int argc, char ** argv,
/// const int readConfigFiles,
/// const int run,
/// int * ierr);
/// ```
pub fn initialize(argv: Vec<String>, read_config_files: bool, run: bool) -> GResult<()> {
    let c_argc = argv.len() as c_int;
    let c_argv = if c_argc == 0 {
        std::ptr::null_mut()
    } else {
        // 使用 CString 安全的将 Rust String 转换为 C 字符数组
        let cstrings = argv
            .into_iter()
            .map(|s| CString::new(s).map_err(|e| e.into()))
            .collect::<Result<Vec<_>, GError>>()?;
        let mut cchars = cstrings
            .iter()
            .map(|s| s.as_ptr() as *mut c_char)
            .collect::<Vec<_>>();
        cchars.push(std::ptr::null_mut());
        cchars.as_mut_ptr()
    };
    let mut ierr = 0;
    unsafe {
        gmshInitialize(
            c_argc,
            c_argv,
            read_config_files as c_int,
            run as c_int,
            &mut ierr,
        );
        if ierr != 0 {
            return Err(GError::from_log());
        }
    }
    Ok(())
}

/// API: gmsh::isInitialized
///
/// Return 1 if the Gmsh API is initialized, and 0 if not.
/// ```c++
/// GMSH_API int isInitialized();
/// ```
/// ```c
/// GMSH_API int gmshIsInitialized(int * ierr);
/// ```
pub fn is_initialized() -> GResult<bool> {
    let mut ierr = 0;
    unsafe {
        let is_initialized = gmshIsInitialized(&mut ierr);
        if ierr != 0 {
            return Err(GError::from_log());
        }
        Ok(is_initialized != 0)
    }
}

/* Finalize the Gmsh API. This must be called when you are done using the Gmsh
 * API. */
/// ```c++
/// GMSH_API void finalize();
/// ```
/// ```c
/// GMSH_API void gmshFinalize(int * ierr);
/// ```
pub fn finalize() -> GResult<()> {
    let mut ierr = 0;
    unsafe {
        gmshFinalize(&mut ierr);
        if ierr != 0 {
            return Err(GError::from_log());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_is_initialized() {
        assert_eq!(is_initialized(), Ok(false));
    }
}
