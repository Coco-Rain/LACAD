import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.trapezoid(10, 5, 60)
.finalize()
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')