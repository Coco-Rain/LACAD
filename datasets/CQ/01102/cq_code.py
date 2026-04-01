import cadquery as cq

result = (
cq.Workplane()
.box(10, 10, 10, centered=(True, True, False))
.tag("brim")
.faces(">Z")
.workplane()
.box(5, 5, 5, centered=(True, True, False))
)
cq.exporters.export(result, 'GT.stl')