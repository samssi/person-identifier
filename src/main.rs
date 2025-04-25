use opencv::{
    core,
    highgui,
    imgproc,
    objdetect,
    prelude::*,
    videoio,
    imgcodecs,
    Result,
};

fn capture_face() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let mut save_faces = false;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Unable to open default camera!");
    }

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

        let mut gray = Mat::default();
        imgproc::cvt_color(
            &frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT
        )?;

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

        for face in faces {
            imgproc::rectangle(
                &mut frame,
                face,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;

            imgproc::put_text(
                &mut frame,
                "Face",
                core::Point::new(face.x, face.y - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                1,
                imgproc::LINE_AA,
                false,
            )?;

            if save_faces {
                let face_roi = core::Rect::new(face.x, face.y, face.width, face.height);
                let cropped = Mat::roi(&frame, face_roi)?;

                // Resize face to a good standard size, e.g., 128x128
                let mut resized_face = Mat::default();
                imgproc::resize(
                    &cropped,
                    &mut resized_face,
                    core::Size::new(128, 128),
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )?;

                let filename = format!("face_{}.png", chrono::Utc::now().timestamp());
                opencv::imgcodecs::imwrite(&filename, &resized_face, &core::Vector::new())?;

                println!("Saved resized face to {}", filename);
            }
        }

        highgui::imshow("webcam", &frame)?;

        let key = highgui::wait_key(10)?;
        if key == 'q' as i32 {
            break;
        } else if key == 's' as i32 {
            save_faces = true;
        }
    }

    Ok(())
}

fn detect_and_resize_face(input_path: &str, output_path: &str) -> Result<()> {
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
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
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

    if faces.len() == 0 {
        println!("No face detected in image: {}", input_path);
        return Ok(());
    }

    // Take the first detected face
    let face = faces.get(0)?;
    let face_roi = core::Rect::new(face.x, face.y, face.width, face.height);
    let cropped = Mat::roi(&img, face_roi)?;

    // Resize cropped face
    let mut resized_face = Mat::default();
    imgproc::resize(
        &cropped,
        &mut resized_face,
        core::Size::new(128, 128),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // Save resized face
    imgcodecs::imwrite(output_path, &resized_face, &core::Vector::new())?;

    println!("Saved resized face to {}", output_path);
    Ok(())
}

fn compare_faces(path1: &str, path2: &str) -> Result<f64> {
    let img1 = opencv::imgcodecs::imread(path1, opencv::imgcodecs::IMREAD_COLOR)?;
    let img2 = opencv::imgcodecs::imread(path2, opencv::imgcodecs::IMREAD_COLOR)?;

    // Resize both images to same size
    let size = core::Size::new(128, 128);
    let mut img1_resized = Mat::default();
    let mut img2_resized = Mat::default();
    imgproc::resize(&img1, &mut img1_resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;
    imgproc::resize(&img2, &mut img2_resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

    // Subtract images
    let mut diff = Mat::default();
    core::absdiff(&img1_resized, &img2_resized, &mut diff)?;

    // Calculate L2 norm of the difference
    let distance = core::norm(
        &diff,
        core::NORM_L2,
        &core::no_array(),
    )?;

    Ok(distance)
}

fn main() -> Result<()> {
    let captured = "face_1745610753.png";
    let samssi = "samssi-resize.png";
    let jp = "juhapekkam-resize.png";
    //detect_and_resize_face("juhapekkam.png", "juhapekkam-resize.png")?;
    //capture_face()

    let jp_distance = compare_faces(captured, jp)?;
    let samssi_distance = compare_faces(captured, samssi)?;


    println!("jp distance: {}", jp_distance);
    println!("samssi distance: {}", samssi_distance);

    Ok(())
}