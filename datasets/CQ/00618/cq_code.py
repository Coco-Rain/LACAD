import cadquery as cq

result = (
cq.Workplane("XZ")
.sketch()
.circle(10)
.rect(5, 3, angle=45, mode='s', tag="cutout")
.finalize()
.extrude(4)
)
cq.exporters.export(result, 'GT.stl')