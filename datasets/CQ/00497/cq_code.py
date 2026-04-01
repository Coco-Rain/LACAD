import cadquery as cq

result = (
cq.Workplane("front")
.circle(8).extrude(5)
.faces(">Z")
.workplane()
.center(0, 4)
.slot2D(10, 3.5, 45)
.extrude(2, taper=10)
)
cq.exporters.export(result, 'GT.stl')