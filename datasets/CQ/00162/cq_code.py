import cadquery as cq

result = (
cq.Workplane("XY")
.cylinder(5, 10)
.faces(">Z")
.workplane()
.polarArray(radius=8, startAngle=0, angle=360, count=6)
.circle(1)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')