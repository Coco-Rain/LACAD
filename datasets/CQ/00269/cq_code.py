import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.regularPolygon(3, 6, angle=45.0, )
.finalize()
.extrude(10)
)
cq.exporters.export(result, 'GT.stl')