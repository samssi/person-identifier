use opencv::Error;
use crate::detector::run_detector;

mod detector;

fn main() -> Result<(), Error> {
    run_detector("samssi-resize.png")?;
    Ok(())
}