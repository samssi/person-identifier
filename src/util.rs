use opencv::core::{Mat, MatTraitConst};
use opencv::{imgcodecs, imgproc, objdetect};

fn clamp_rect(rect: opencv::core::Rect, image: &Mat) -> Option<opencv::core::Rect> {
    if rect.width <= 0 || rect.height <= 0 {
        return None;
    }

    let x = rect.x.max(0).min(image.cols() - 1);
    let y = rect.y.max(0).min(image.rows() - 1);
    let width = (rect.width).min(image.cols() - x);
    let height = (rect.height).min(image.rows() - y);

    if width <= 0 || height <= 0 {
        None
    } else {
        Some(opencv::core::Rect::new(x, y, width, height))
    }
}

fn is_valid_face(rect: &opencv::core::Rect) -> bool {
    let min_size = 40;
    let max_aspect_ratio = 1.5;

    if rect.width < min_size || rect.height < min_size {
        return false;
    }

    let aspect_ratio = rect.width as f32 / rect.height as f32;
    aspect_ratio > (1.0 / max_aspect_ratio) && aspect_ratio < max_aspect_ratio
}

fn compute_histogram(face: &Mat) -> opencv::Result<Mat> {
    let mut hist = Mat::default();
    let hist_size = opencv::core::Vector::from(vec![256]);
    let ranges = opencv::core::Vector::from(vec![0f32, 256f32]);
    let channels = opencv::core::Vector::from(vec![0]);

    imgproc::calc_hist(
        face,
        &channels,
        &opencv::core::no_array(),
        &mut hist,
        &hist_size,
        &ranges,
        false,
    )?;

    let mut normalized_hist = Mat::default();
    opencv::core::normalize(
        &hist,
        &mut normalized_hist,
        0.0,
        1.0,
        opencv::core::NORM_MINMAX,
        -1,
        &opencv::core::no_array(),
    )?;

    Ok(normalized_hist)
}

fn detect_and_resize_face(input_path: &str, output_path: &str) -> opencv::Result<()> {
    // Load Haar Cascade for face detection
    let mut face_cascade = objdetect::CascadeClassifier::new(
        "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )?;

    // Read input image
    let img = imgcodecs::imread(input_path, imgcodecs::IMREAD_COLOR)?;

    // Convert to grayscale
    let mut gray = Mat::default();
    imgproc::cvt_color(
        &img,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Detect faces
    let mut faces = opencv::core::Vector::<opencv::core::Rect>::new();
    face_cascade.detect_multi_scale(
        &gray,
        &mut faces,
        1.1,
        3,
        0,
        opencv::core::Size::new(30, 30),
        opencv::core::Size::new(0, 0),
    )?;

    if faces.len() == 0 {
        println!("No face detected in image: {}", input_path);
        return Ok(());
    }

    // Take the first detected face
    let face = faces.get(0)?;
    let safe_face_roi = match crate::clamp_rect(face, &img) {
        Some(r) => r,
        None => {
            println!("Detected face is invalid after clamping.");
            return Ok(());
        }
    };
    let cropped = Mat::roi(&img, safe_face_roi)?;

    // Resize cropped face
    let mut resized_face = Mat::default();
    imgproc::resize(
        &cropped,
        &mut resized_face,
        opencv::core::Size::new(128, 128),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // Save resized face
    imgcodecs::imwrite(output_path, &resized_face, &opencv::core::Vector::new())?;

    println!("Saved resized face to {}", output_path);
    Ok(())
}
