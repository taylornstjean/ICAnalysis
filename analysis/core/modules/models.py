from icecube import icetray, dataclasses
from icecube.icetray import I3Bool


class I3Alerts(icetray.I3Module):
    def __init__(self, context):
        super(I3Alerts, self).__init__(context)  # Initialize the base class

    def Physics(self, frame, **kwargs):
        """ Called for each physics event. """
        alerts = list(frame["AlertNamesPassed"])
        hese = "HESE" in alerts
        frame["HESEBool"] = I3Bool(hese)

        # Pass the frame along the pipeline
        self.PushFrame(frame)

