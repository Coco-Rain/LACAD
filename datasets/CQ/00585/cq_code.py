import cadquery as cq

result = (
cq.Workplane("XY")
.cylinder(10, 5)
.faces("+Z")
.workplane()
.center(0, -3)
.rect(6, 2, centered=False)
.extrude(-1.5)
)
cq.exporters.export(result, 'GT.stl')