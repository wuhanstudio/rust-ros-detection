use image::{ImageBuffer, RgbImage};

// use image::Rgb;

use rosrust;

use anyhow::Result;

use std::sync::{Arc, Mutex};

use darknet::{Network, BBox};
use std::slice;
use std::{
    path::Path,
};

use darknet_sys;


const LABEL_PATH: &'static str = "coco.names";
const CFG_PATH: &'static str = "yolov3-tiny.cfg";
const WEIGHTS_FILE_NAME: &'static str = "yolov3-tiny.weights";

// const OUTPUT_DIR: &'static str = "./output";

const OBJECTNESS_THRESHOLD: f32 = 0.6;
const CLASS_PROB_THRESHOLD: f32 = 0.6;

#[show_image::main]
fn main() -> Result<(), String> {

    // Load weights file
    let weights_path = Path::new(WEIGHTS_FILE_NAME);

    // Load network & labels
    let object_labels = std::fs::read_to_string(LABEL_PATH).unwrap()
        .lines()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let net = Arc::new(Mutex::new(Network::load(CFG_PATH, Some(weights_path), false).unwrap()));

    let window = show_image::create_window("image", Default::default()).map_err(|e| e.to_string())?;

    // Initialize node
    rosrust::init("listener");

    // Create subscriber
    // The subscriber is stopped when the returned object is destroyed
    let _subscriber_raii = rosrust::subscribe(
        "/usb_cam/image_raw",
        1,
        move |v: rosrust_msg::sensor_msgs::Image| {
            // Run object detection

            let width = v.width as i32;
            let height = v.height as i32;
            let channel = 3 as i32;

            let dimg = unsafe { darknet_sys::make_image(width, height, channel) };
            let slice = unsafe { slice::from_raw_parts_mut(dimg.data, (width*height*channel) as usize) }; 
            for i in 0..((width*height*3) as usize) {
                slice[i] = v.data[i] as f32 / 255.0;
            }
            // let d_img = unsafe { darknet_sys::resize_image(dimg, 416, 416) };
            let img = darknet::Image {
                image: dimg
            };
            let detections = net.lock().unwrap().predict(&img, 0.25, 0.5, 0.45, true);

            // show results
            detections
            .iter()
            .filter(|det| det.objectness() > OBJECTNESS_THRESHOLD)
            .flat_map(|det| {
                det.best_class(Some(CLASS_PROB_THRESHOLD))
                    .map(|(class_index, prob)| (det, prob, &object_labels[class_index]))
            })
            .enumerate()
            .for_each(|(_, (det, prob, label))| {
                let bbox = det.bbox();
                let BBox { x, y, w, h } = bbox;

                // Save image
                // let image_path =
                // Path::new(OUTPUT_DIR).join(format!("{}-{}-{:2.2}.jpg", index, label, prob * 100.0));
                //     img
                //     .crop_bbox(bbox)
                //     .to_image_buffer::<Rgb<u8>>()
                //     .unwrap()
                //     .save(image_path)
                //     .unwrap();

                // print result
                println!(
                    "{}\t{:.2}%\tx: {}\ty: {}\tw: {}\th: {}",
                    label,
                    prob * 100.0,
                    x,
                    y,
                    w,
                    h
                );
            });

            // Callback for handling received messages
            let mut image: RgbImage = ImageBuffer::new(img.image.w as u32, img.image.h as u32);

            // Constructing image from darknet image
            for i in 0..img.image.w {
                for j in 0..img.image.h{
                    let i_usize: usize = i as usize;
                    let j_usize: usize = j as usize;
                    *image.get_pixel_mut(i as u32, j as u32) = image::Rgb([ (slice[(i_usize + j_usize*(width as usize))*(channel as usize) + 0] * 255.0) as u8, (slice[(i_usize + j_usize*(width as usize))*(channel as usize) + 1] * 255.0) as u8, (slice[(i_usize + j_usize*(width as usize))*(channel as usize) + 2] * 255.0) as u8 ]);
                }
            }

            window.set_image("ROS Image", image.clone()).unwrap();
        },
    )
    .unwrap();

    // Block the thread until a shutdown signal is received
    rosrust::spin();

    Ok(())
}
