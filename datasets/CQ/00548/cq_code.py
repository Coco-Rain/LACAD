import cadquery as cq

result = (
cq.Workplane("XY")
.box(5, 5, 2)
.faces(">Z")
.workplane()
.center(1, 1)
.circle(0.5)
.extrude(0.5)
)
number_of_objects = result.size()
print(f"Number of objects on the stack: {number_of_objects}")
cq.exporters.export(result, 'GT.stl')