import cadquery as cq

result = (
cq.Workplane("YZ")
.box(20, 20, 2)
.rotateAboutCenter((0, 1, 0), 45)
)
cq.exporters.export(result, 'GT.stl')