import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 3)
.faces("<Z")
.workplane(invert=True)
.center(10, 10)
.cskHole(diameter=6, cskDiameter=10, cskAngle=90)
)
cq.exporters.export(result, 'GT.stl')