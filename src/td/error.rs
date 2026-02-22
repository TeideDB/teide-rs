use crate::Error;

#[derive(Debug, Clone)]
pub enum TdError {
    Parse(String),
    Compile(String),
    Runtime(String),
    Engine(Error),
}

impl std::fmt::Display for TdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TdError::Parse(msg) => write!(f, "parse error: {msg}"),
            TdError::Compile(msg) => write!(f, "compile error: {msg}"),
            TdError::Runtime(msg) => write!(f, "runtime error: {msg}"),
            TdError::Engine(err) => write!(f, "engine error: {err}"),
        }
    }
}

impl std::error::Error for TdError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TdError::Engine(e) => Some(e),
            _ => None,
        }
    }
}

impl From<Error> for TdError {
    fn from(err: Error) -> Self {
        TdError::Engine(err)
    }
}

pub type TdResult<T> = Result<T, TdError>;
