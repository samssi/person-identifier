use opencv::{
    prelude::*,
    videoio,
    highgui,
    Result,
};

fn main() -> Result<()> {
    // Open the default camera (index 0)
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is default cam
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Unable to open default camera!");
    }

    // Create a window
    highgui::named_window("webcam", highgui::WINDOW_AUTOSIZE)?;

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;

        if frame.size()?.width > 0 {
            highgui::imshow("webcam", &frame)?;
        }

        // Break on 'q' key
        let key = highgui::wait_key(10)?;
        if key == 'q' as i32 {
            break;
        }
    }

    Ok(())
}