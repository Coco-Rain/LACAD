import cadquery as cq

result = (
cq.Workplane("XY")
.rect(10, 10)
.rotateAboutCenter((0, 0, 1), 45)
.extrude(5)
)
cq.exporters.export(result, 'GT.stl')