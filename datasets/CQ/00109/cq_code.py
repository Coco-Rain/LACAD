import cadquery as cq

result = (
cq.Workplane("XY")
.circle(5)
.extrude(2)
.faces(">Z").workplane()
.rect(3, 3)
.cutBlind(-1)
)
all_objects = result.all()
print(all_objects)
cq.exporters.export(result, 'GT.stl')