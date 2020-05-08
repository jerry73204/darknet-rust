use darknet::{Detection, Image, Network};
use failure::Fallible;
use image::{ImageBuffer, Rgb};
use itertools::{izip, Itertools};
use std::convert::TryFrom;

const LABEL_PATH: &'static str = "./darknet/data/coco.names";
const IMAGE_PATH: &'static str = "./darknet/data/person.jpg";
const CFG_PATH: &'static str = "./darknet/cfg/yolov3-tiny.cfg";
const WEIGHTS_PATH: &'static str = "./yolov3-tiny.weights";
const OBJECTNESS_THRESHOLD: f32 = 0.8;
const CLASS_PROB_THRESHOLD: f32 = 0.5;

fn main() -> Fallible<()> {
    // Load network & labels
    let object_labels = std::fs::read_to_string(LABEL_PATH)?
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    let mut net = Network::load(CFG_PATH, Some(WEIGHTS_PATH), false)?;
    let img = Image::open(IMAGE_PATH)?;

    // Run object detection
    let detections = net.predict(&img, 0.25, 0.5, 0.45, true);

    let get_max_prob = |det: &Detection| {
        izip!(det.probabilities().iter().cloned(), object_labels.iter())
            .fold1(|prev, curr| {
                let (prev_prob, _) = prev;
                let (curr_prob, _) = curr;
                if curr_prob > prev_prob {
                    curr
                } else {
                    prev
                }
            })
            .unwrap()
    };

    // let buffer: ImageBuffer<Rgb<f32>, Vec<_>> = TryFrom::try_from(&img)?;

    detections
        .iter()
        .filter(|det| det.objectness() > OBJECTNESS_THRESHOLD)
        .flat_map(|det| {
            let (max_prob, label) = get_max_prob(&det);

            if max_prob > CLASS_PROB_THRESHOLD {
                Some((det, max_prob, label))
            } else {
                None
            }
        })
        .for_each(|(det, prob, label)| {
            let bbox = det.bbox();
            // let BBox {x, y, w, h} = det.bbox();
            // let rect = Rect {};
            // imageproc::drawing::draw_hollow_rect(&buffer, );
            println!("{}\t{:.2}%\t{:?}", label, prob * 100.0, bbox);
        });

    Ok(())
}
