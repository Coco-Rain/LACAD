import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 10)
.sketch()
.circle(2).reset()
.circle(1)
.finalize()
.extrude(5)
)
cq.exporters.export(result, 'GT.stl')