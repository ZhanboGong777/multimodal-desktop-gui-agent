"""Four unchanged OCR arms on one preannotated saved real frame; no desktop API."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASE = ROOT / "Document/Week6/results/full_requirements_20261028"
DATA = BASE / "perception_fixture_001"
OUT = BASE / "perception_external_real_001"
ARMS = ["native_eng_subprocess", "native_bilingual_subprocess",
        "screen_bilingual_subprocess", "screen_bilingual_persistent"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    with Path(path).open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2)+"\n").encode("utf-8"))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def timing(rows):
    values = sorted(row["full_ms"] for row in rows)
    return {"n": len(values), "median_ms": statistics.median(values),
            "p95_nearest_rank_ms": values[math.ceil(.95*len(values))-1], "max_ms": max(values)} if values else None


if __name__ == "__main__":
    from gui_agent.perception.ocr import TesseractOCREngine
    from gui_agent.perception.ui_detection import detect_ui_candidates

    freeze = read(BASE/"perception_source_freeze_001.json")
    assert all(sha(ROOT/path) == digest for path,digest in freeze['source_pins'].items())
    assert all(sha(path) == digest for path,digest in freeze['runtime_pins'].items())
    gt_path = OUT/"ground_truth_frozen_001.json"
    gt = read(gt_path)
    assert sha(gt['copied_path']) == gt['copied_sha256'] == gt['source_sha256']
    assert sha(gt['source_path']) == gt['source_sha256']
    image_path = Path(gt['copied_path'])
    output = OUT/"paired_measurement_001"
    output.mkdir(exist_ok=False)
    save(output/'source_freeze.json', {'created_utc':datetime.now(UTC).isoformat(),
            'script_sha256':sha(__file__),'source_freeze_sha256':sha(BASE/'perception_source_freeze_001.json'),
            'gt_sha256':sha(gt_path),'gt_created_utc':gt['created_utc'],
            'method':'3 fresh-engine calls + 1 retained warmup +10 rotated paired repeats per arm, complete uncropped source. ROI only scores; no model/GUI/capture/input.',
            'no_speed_or_accuracy_superiority_claim_preregistered_for_this_single_frame':True})
    baseline = module('gui_agent.perception._external_frozen_cli', DATA/'baseline_ocr.py').TesseractOCREngine
    scoring = module('_external_frozen_metrics', ROOT/'scripts/week6_benchmark_accuracy.py')
    records = []

    def factory(arm):
        options = {'default_languages':['en'] if arm == ARMS[0] else ['en','zh'],
                   'tessdata_dir':Path('D:/week6-full-ocr/tessdata')}
        if arm.startswith('screen'):
            options.update(profile='screen',scale=2.0)
        if arm == ARMS[3]:
            return TesseractOCREngine(**options,transport='persistent',
                    library_path='C:/Program Files/Tesseract-OCR/libtesseract-5.dll')
        return baseline(**options)

    def within(element):
        b = element.bounding_box
        roi = gt['scoring_roi']
        return b.left >= roi[0] and b.top >= roi[1] and b.right <= roi[2] and b.bottom <= roi[3]

    def measure(arm, phase, repeat, engine=None):
        owned = engine is None
        start = time.perf_counter()
        result = {'arm':arm,'phase':phase,'repeat':repeat,'image_sha256':gt['copied_sha256'],
                  'started_perf_counter':start,'input_full_size':gt['size']}
        try:
            if engine is None:
                engine = factory(arm)
            created = time.perf_counter()
            with Image.open(image_path) as raw:
                image = raw.convert('RGB')
            decoded = time.perf_counter()
            original_pixels = image.tobytes()
            recognized = engine.recognize(image)
            ocr_done = time.perf_counter()
            candidates = detect_ui_candidates(image,exclude=[e.bounding_box for e in recognized.elements])
            end = time.perf_counter()
            assert original_pixels == image.tobytes()
            roi_predictions = [item for item in recognized.elements if within(item)]
            roi_candidates = [item for item in candidates if within(item)]
            result.update(status='completed',ended_perf_counter=end,full_ms=(end-start)*1000,
                          engine_creation_ms=(created-start)*1000,decode_ms=(decoded-created)*1000,
                          ocr_ms=(ocr_done-decoded)*1000,ui_ms=(end-ocr_done)*1000,
                          predictions=[e.model_dump(mode='json') for e in recognized.elements],
                          ui_candidates=[e.model_dump(mode='json') for e in candidates],
                          roi_metrics=scoring.measure(gt['labels'],roi_predictions),
                          roi_control_metrics=scoring.control_metrics(gt['controls'],roi_candidates),
                          errors=recognized.errors,full_frame_precision=None,full_frame_recall=None)
        except Exception:  # noqa: BLE001 -- retain each independent failed arm measurement.
            result.update(status='measurement_error',traceback=traceback.format_exc(),
                          observed_elapsed_ms=(time.perf_counter()-start)*1000)
        finally:
            if owned and engine is not None and hasattr(engine,'close'):
                engine.close()
        save(output/f'{phase}_{repeat:02d}_{arm}.json',result)
        records.append(result)

    for repeat in range(3):
        for arm in ARMS[repeat:]+ARMS[:repeat]:
            measure(arm,'cold',repeat)
    engines = {arm:factory(arm) for arm in ARMS}
    try:
        for arm in ARMS:
            measure(arm,'warmup',0,engines[arm])
        for repeat in range(10):
            rotation = repeat%4
            order = ARMS[rotation:]+ARMS[:rotation]
            if repeat%2:
                order.reverse()
            for arm in order:
                measure(arm,'warm',repeat,engines[arm])
    finally:
        for engine in engines.values():
            if hasattr(engine,'close'):
                engine.close()
    success = [r for r in records if r['status']=='completed']
    exact = []
    for candidate in [r for r in success if r['arm']==ARMS[3]]:
        other = [r for r in success if r['arm']==ARMS[2] and r['phase']==candidate['phase'] and r['repeat']==candidate['repeat']]
        exact.append({'phase':candidate['phase'],'repeat':candidate['repeat'],'equal':len(other)==1 and
                      all(candidate[k]==other[0][k] for k in ('predictions','ui_candidates','roi_metrics','roi_control_metrics'))})
    values = {phase:{arm:timing([r for r in success if r['phase']==phase and r['arm']==arm])
                      for arm in ARMS} for phase in ('cold','warmup','warm')}
    metrics = {r['arm']:{'text':r['roi_metrics'],'controls':r['roi_control_metrics']}
               for r in success if r['phase']=='warm' and r['repeat']==0}
    source_ok = all(sha(ROOT/p)==h for p,h in freeze['source_pins'].items())
    runtime_ok = all(sha(p)==h for p,h in freeze['runtime_pins'].items())
    result = {'schema':'week6.full_perception.real_saved_frame_result.v1','created_utc':datetime.now(UTC).isoformat(),
              'source_sha256':gt['copied_sha256'],'gt_sha256':sha(gt_path),'records':len(records),
              'timings':values,'roi_metrics_unique_frame':metrics,'equivalence':exact,
              'all_errors':[r for r in records if r['status']!='completed' or r.get('errors')],
              'source_unchanged':source_ok,'runtime_unchanged':runtime_ok,'inputs_unchanged':sha(image_path)==gt['copied_sha256'] and sha(gt_path)==read(output/'source_freeze.json')['gt_sha256'],
              'screen_full_predictions_exact':len(exact)==14 and all(r['equal'] for r in exact),
              'full_frame_precision':None,'full_frame_recall':None,'roi':gt['scoring_roi'],
              'limits':'One easy English real frame; ROI5 labels/2 controls only. Whole uncropped PNG used for full processing timing; unannotated notification retained. No improvement expected from labels, no code/parameter changes, no execution-success claim.',
              'no_gpu_frameworks_imported':not any(sys.modules.get(x) is not None for x in ('torch','paddle')),
              'omp_environment':{key:os.environ.get(key) for key in ('OMP_THREAD_LIMIT','OMP_NUM_THREADS')}}
    save(output/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
