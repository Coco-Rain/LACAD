import cadquery as cq

result = (
cq.Workplane("XZ")
.sketch()
.ellipse(6, 4)
.rect(12, 6)
.finalize()
.extrude(4)
)
cq.exporters.export(result, 'GT.stl')