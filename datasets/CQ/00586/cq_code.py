import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.regularPolygon(5, 6)
.finalize()
.extrude(10)
)
cq.exporters.export(result, 'GT.stl')