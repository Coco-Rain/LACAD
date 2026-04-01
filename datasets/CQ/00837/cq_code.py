import cadquery as cq

result = (
cq.Workplane("XY")
.polyline([(0, 0), (5, 0), (0, 5)])
.close()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')