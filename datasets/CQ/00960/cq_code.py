import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.circle(5)
.rect(3, 3)
.finalize()
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')