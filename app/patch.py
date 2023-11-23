import ultralytics.engine.trainer
import ultralytics.utils.checks


def check_amp(model) -> bool:  # type: ignore
    return True


# enforce AMP
ultralytics.engine.trainer.check_amp = check_amp
