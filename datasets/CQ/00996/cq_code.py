import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 1)
.faces(">Z")
.workplane()
.center(2, 2)
.circle(1)
.extrude(1)
)
number_of_objects = result.size()
print(f"Number of objects on the stack: {number_of_objects}")
cq.exporters.export(result, 'GT.stl')