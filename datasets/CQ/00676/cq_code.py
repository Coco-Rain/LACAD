import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 1)
.faces(">Z")
.workplane()
.center(2, 2)
.circle(1)
.extrude(5)
)
val_result = result.val()
print(val_result)
cq.exporters.export(result, 'GT.stl')