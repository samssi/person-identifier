use std::path::Path;
use opencv::{objdetect, videoio, imgcodecs, Error, imgproc, highgui};
use opencv::boxed_ref::BoxedRef;
use opencv::core::{absdiff, no_array, norm, Mat, MatTraitConst, Rect, ToInputArray, Vector, NORM_L2};
use opencv::prelude::CascadeClassifierTrait;
use opencv::videoio::{VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst};

fn image_as_matrix(image: &str) -> Result<Mat, Error> {
    let reference_image_matrix = imgcodecs::imread(image, opencv::imgcodecs::IMREAD_COLOR)?;
    Ok(reference_image_matrix)
}

fn colored_image_to_black_and_white(original_image_matrix: &Mat) -> Result<Mat, Error> {
    let mut bw_image_matrix = Mat::default();

    imgproc::cvt_color(
        original_image_matrix,
        &mut bw_image_matrix,
        imgproc::COLOR_BGR2GRAY,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    Ok(bw_image_matrix)
}

fn resize_image(image_matrix: impl ToInputArray) -> Result<Mat, Error> {
    let mut reference_resized = Mat::default();
    imgproc::resize(
        &image_matrix,
        &mut reference_resized,
        opencv::core::Size::new(128, 128),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    Ok(reference_resized)
}

fn filename_from_image_path(image_path: &str) -> &str {
    Path::new(image_path)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap_or("Unknown")
}

fn try_to_start_video_capture() -> Result<VideoCapture, Error> {
    let cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !VideoCapture::is_opened(&cam)? {
        panic!("failed to open capture device")
    }
    Ok(cam)
}

fn add_identification_rectangle_to_image(image: &Mat, face: Rect) -> Result<Mat, Error> {
    let mut image_with_rectangle = image.clone();

    imgproc::rectangle(
        &mut image_with_rectangle,
        face,
        opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;

    Ok(image_with_rectangle)
}

fn detect_face_with_haar_cascade(image: &Mat) -> Result<Vector<Rect>, Error> {
    let mut face_cascade = objdetect::CascadeClassifier::new(
        "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )?;

    let mut faces_ref = Vector::<Rect>::new();
    face_cascade.detect_multi_scale(
        image,
        &mut faces_ref,
        1.1,
        3,
        0,
        opencv::core::Size::new(30, 30),
        opencv::core::Size::new(0, 0),
    )?;

    Ok(faces_ref)
}

fn compare_faces(a: &Mat, b: &Mat) -> Result<f64, Error> {
    let mut diff = Mat::default();
    absdiff(&a, &b, &mut diff)?;

    let result = norm(&diff, NORM_L2, &no_array())?;
    Ok(result)
}

pub fn add_label_to_face(image: &Mat, face: &Rect, distance: f64, filename: &str) -> Result<Mat, Error> {
    let mut labeled_image = image.clone();

    let label = if distance < 20000.0 {
        format!("{}, distance: {:.2}", filename, distance)
    } else {
        format!("Unknown, distance: {:.2}", distance)
    };

    imgproc::put_text(
        &mut labeled_image,
        &label,
        opencv::core::Point::new(face.x, face.y - 10),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(labeled_image)
}

pub fn to_bw_image(image: &Mat) -> Result<Mat, Error> {
    let mut bw_image = Mat::default();
    imgproc::cvt_color(
        &image,
        &mut bw_image,
        imgproc::COLOR_BGR2GRAY,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    Ok(bw_image)
}

pub fn run_detector(reference_image_path: &str) -> Result<(), Error> {
    highgui::named_window("webcam", highgui::WINDOW_AUTOSIZE)?;
    let mut cam = try_to_start_video_capture()?;
    let loaded_reference_image = imgcodecs::imread(reference_image_path, opencv::imgcodecs::IMREAD_GRAYSCALE)?;
    let reference_image_faces = detect_face_with_haar_cascade(&loaded_reference_image)?;
    if reference_image_faces.is_empty() {
        panic!("No valid reference image found!")
    }

    let reference_image_face = reference_image_faces.get(0)?;
    let reference_image_ref = Mat::roi(&loaded_reference_image, reference_image_face)?;
    let reference_image = resize_image(reference_image_ref)?;

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;

        if frame.size()?.width == 0 {
            continue;
        }

        let bw_frame = colored_image_to_black_and_white(&frame)?;
        let bw_image_resized = resize_image(bw_frame)?;
        let mut resulting_frame = frame.clone();

        let faces = detect_face_with_haar_cascade(&bw_image_resized)?;
        for face in faces {
            // Add box where identification happened to the original frame
            let frame_with_debug_box = add_identification_rectangle_to_image(&frame, face)?;

            let face_roi = Rect::new(face.x, face.y, face.width, face.height);
            let person_face = Mat::roi(&frame_with_debug_box, face_roi)?;

            let person_face_resized = resize_image(person_face)?;
            let person_face_resized_bw = to_bw_image(&person_face_resized)?;
            let distance = compare_faces(&person_face_resized_bw, &reference_image)?;

            resulting_frame = add_label_to_face(&frame_with_debug_box, &face, distance, reference_image_path)?;
        }

        highgui::imshow("webcam", &resulting_frame)?;

        let key = highgui::wait_key(10)?;
        if key == 'q' as i32 {
            break;
        }
    }
    Ok(())
}