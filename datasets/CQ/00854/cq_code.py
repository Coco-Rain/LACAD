import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.circle(5)
.finalize()
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')