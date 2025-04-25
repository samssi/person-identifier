use opencv::{
    core,
    highgui,
    imgproc,
    objdetect,
    prelude::*,
    videoio,
    Result,
};

fn main() -> Result<()> {
    // Open webcam
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Unable to open default camera!");
    }

    // Load Haar Cascade for face detection
    let mut face_cascade = objdetect::CascadeClassifier::new(
        "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )?;

    highgui::named_window("webcam", highgui::WINDOW_AUTOSIZE)?;

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            continue;
        }

        // Convert frame to grayscale
        let mut gray = Mat::default();
        imgproc::cvt_color(
            &frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT
        )?;

        // Detect faces
        let mut faces = core::Vector::<core::Rect>::new();
        face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            3,
            0,
            core::Size::new(30, 30),
            core::Size::new(0, 0),
        )?;

        // Draw rectangles around faces
        for face in faces {
            imgproc::rectangle(
                &mut frame,
                face,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0), // green
                2,
                imgproc::LINE_8,
                0,
            )?;

            imgproc::put_text(
                &mut frame,
                "Face",
                core::Point::new(face.x, face.y - 10), // Slightly above the rectangle
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5, // font scale
                core::Scalar::new(0.0, 255.0, 0.0, 0.0), // same green color
                1, // thickness
                imgproc::LINE_AA,
                false,
            )?;
        }

        highgui::imshow("webcam", &frame)?;

        let key = highgui::wait_key(10)?;
        if key == 'q' as i32 {
            break;
        }
    }

    Ok(())
}