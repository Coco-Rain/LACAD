import cadquery as cq

result = (
cq.Workplane("XZ")
.sketch()
.rect(20, 20)
.rect(10, 10, mode='s')
.circle(3, mode='a')
.clean()
.finalize()
.extrude(10)
)
cq.exporters.export(result, 'GT.stl')