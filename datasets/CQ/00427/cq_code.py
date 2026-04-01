import cadquery as cq

result = (
cq.Workplane("XY")
.box(15, 15, 3)
.faces(">Z")
.workplane()
.center(7.5, 7.5)
.cskHole(diameter=5, cskDiameter=8, cskAngle=82)
)
cq.exporters.export(result, 'GT.stl')