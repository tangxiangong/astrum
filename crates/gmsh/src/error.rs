use crate::logger::get_last_error;

#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum GError {
    #[error("Gmsh error: {0}")]
    GmshError(String),
    #[error("Transfer string error: {0}")]
    TransferStringError(#[from] std::ffi::NulError),
    #[error("空指针")]
    NullPointerError,
    #[error("无效状态")]
    InvalidState(String),
    #[error("API错误")]
    ApiError(String),
}

pub type GResult<T> = Result<T, GError>;

impl GError {
    pub fn from_log() -> Self {
        match get_last_error() {
            Ok(error) => GError::GmshError(error),
            Err(e) => e,
        }
    }
}
    