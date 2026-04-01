import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.circle(10)
.circle(5)
.clean()
.finalize()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')